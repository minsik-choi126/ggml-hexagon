// test-npu.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <climits>

#include "ggml-impl.h"     // ggml_time_us()
#include "ggml-hexagon.h"  // qnn_instance / qnn_interface 래퍼 포함

// QNN SDK
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnInterface.h"
#include "QnnSystemInterface.h"
#include "QnnOpPackage.h"  // QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_ELEMENT_WISE_ADD

#ifndef MAX_ALIGNMENT
#define MAX_ALIGNMENT 64
#endif
#ifndef WARMUP
#define WARMUP 5
#endif
#ifndef ITERATIONS
#define ITERATIONS 10
#endif

// 간단 에러 체크
#define CHECK_QNN_OK(expr)                                      \
  do {                                                          \
    Qnn_ErrorHandle_t _e = (expr);                              \
    if (_e != QNN_SUCCESS) {                                    \
      std::printf("QNN error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
      return 1;                                                 \
    }                                                           \
  } while (0)

static void* align_with_offset(void* base, size_t bytes, int offset) {
    void*  p = base;
    size_t space = bytes;
    void* aligned = std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, p, space);
    if (!aligned) return nullptr;
    return static_cast<char*>(aligned) + offset;
}

static void generate_fp32(float offset, size_t n, float* dst) {
    for (size_t i = 0; i < n; ++i) dst[i] = 0.1f + 2.0f * std::cos(float(i) + offset);
}

static float compute_rmse(const float* ref, const float* out, size_t n) {
    if (n == 0) return 0.f;
    long double s = 0;
    for (size_t i = 0; i < n; ++i) {
        long double d = (long double)ref[i] - (long double)out[i];
        s += d*d;
    }
    return (float)std::sqrt(s / (long double)n);
}

// QNN 텐서 v1 채우기
static Qnn_Tensor_t make_tensor_v1(const char* name,
                                   Qnn_TensorType_t type,
                                   Qnn_DataType_t  dtype,
                                   uint32_t rank,
                                   const uint32_t* dims,
                                   void* client_buf,
                                   uint32_t byte_size) {
    // 이름/차원 메모리는 호출자가 수명 유지
    Qnn_Tensor_t t{};
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.id   = 0;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType   = dtype;
    // quant 파라미터는 비설정
    t.v1.quantizeParams.encodingDefinition   = QNN_DEFINITION_UNDEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t.v1.quantizeParams.scaleOffsetEncoding.scale = 0.0f;
    t.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    t.v1.rank = rank;
    t.v1.dimensions = const_cast<uint32_t*>(dims);
    // RPC 미사용 경로: RAW client buffer
    t.v1.memType   = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = { client_buf, byte_size };
    return t;
}

// 단일 그래프 빌드(Add) + 실행/측정
static int build_and_bench_add_graph(
        qnn_instance& inst,
        const std::vector<uint32_t>& shape,
        Qnn_DataType_t dtype,
        const void* in_buf,
        const void* zero_buf,
        void* out_buf,
        size_t elem_count,
        int alignment_offset,
        int64_t& min_us,
        double& avg_us)
{
    const qnn_interface&     qnn  = inst.get_qnn_interface();
    const QNN_INTERFACE_VER_TYPE& qnn_raw = inst.get_qnn_raw_interface();

    const uint32_t rank = (uint32_t)shape.size();
    // 이름/차원 수명 보장
    std::string in_name   = "in";
    std::string zer_name  = "zero";
    std::string out_name  = "out";
    std::vector<uint32_t> dims(shape.begin(), shape.end());

    const uint32_t elem_size = (dtype == QNN_DATATYPE_FLOAT_32) ? 4u : 1u;
    const uint32_t nbytes    = (uint32_t)(elem_count * elem_size);

    // 그래프 생성 (qnn_instance가 backend/context 내부 준비)
    {
        // VTMC/HVX 등은 필요시 조절; 여기선 기본값
        const std::string gname = "bench_add";
        const size_t vtcm_mb    = 0;   // 0: default
        const size_t hvx_thr    = 0;   // 0: default
        if (inst.init_qnn_graph(gname, HEXAGON_BACKEND_QNNNPU, vtcm_mb, hvx_thr) != 0) {
            std::printf("init_qnn_graph failed\n");
            return 1;
        }
    }
    Qnn_GraphHandle_t graph = inst.get_qnn_graph_handle();
    if (!graph) { std::printf("graph handle null\n"); return 1; }

    // 텐서 생성 & 그래프에 등록
    Qnn_Tensor_t in_t  = make_tensor_v1(in_name.c_str(),  QNN_TENSOR_TYPE_APP_WRITE, dtype, rank, dims.data(),
                                        const_cast<void*>(in_buf), nbytes);
    Qnn_Tensor_t z_t   = make_tensor_v1(zer_name.c_str(), QNN_TENSOR_TYPE_STATIC,    dtype, rank, dims.data(),
                                        const_cast<void*>(zero_buf), nbytes);
    Qnn_Tensor_t out_t = make_tensor_v1(out_name.c_str(), QNN_TENSOR_TYPE_APP_READ,  dtype, rank, dims.data(),
                                        out_buf, nbytes);

    CHECK_QNN_OK(qnn_raw.tensorCreateGraphTensor(graph, &in_t));
    CHECK_QNN_OK(qnn_raw.tensorCreateGraphTensor(graph, &z_t));
    CHECK_QNN_OK(qnn_raw.tensorCreateGraphTensor(graph, &out_t));

    // 노드(Add) 추가
    Qnn_Tensor_t inputs[]  = { in_t, z_t };
    Qnn_Tensor_t outputs[] = { out_t };

    // 연산 설정 (패키지/타입은 QTI AISW 표준 elementwise add)
    Qnn_OpConfig_t opcfg{};
    {
        static char op_name[64];
        std::snprintf(op_name, sizeof(op_name), "op_add_%u", (unsigned)rank);
        Qnn_OpConfigV1_t v1{};
        v1.name         = op_name;
        v1.packageName  = QNN_OP_PACKAGE_NAME_QTI_AISW;
        v1.type         = QNN_OP_ELEMENT_WISE_ADD;
        v1.numOfParams  = 0;
        v1.params       = nullptr;
        v1.numOfInputs  = 2;
        v1.inputs       = inputs;
        v1.numOfOutputs = 1;
        v1.outputs      = outputs;
        opcfg.version   = QNN_OPCONFIG_VERSION_1;
        opcfg.v1        = v1;
    }
    CHECK_QNN_OK(qnn_raw.graphAddNode(graph, opcfg));

    // finalize
    CHECK_QNN_OK(qnn_raw.graphFinalize(graph, nullptr, nullptr));

    // 워밍업
    for (int i = 0; i < WARMUP; ++i) {
        CHECK_QNN_OK(qnn_raw.graphExecute(graph, nullptr /* executeFlags */));
    }

    // 측정
    int64_t total_us = 0;
    min_us = INT64_MAX;
    for (int i = 0; i < ITERATIONS; ++i) {
        const int64_t t0 = ggml_time_us();
        Qnn_ErrorHandle_t e = qnn_raw.graphExecute(graph, nullptr);
        const int64_t t1 = ggml_time_us();
        if (e != QNN_SUCCESS) {
            std::printf("    EXEC_FAIL (%d)\n", (int)e);
            return 2;
        }
        const int64_t dt = t1 - t0;
        total_us += dt;
        if (dt < min_us) min_us = dt;
    }
    avg_us = (double)total_us / (double)ITERATIONS;

    // 그래프 핸들 자체 해제 API가 노출되어 있지 않으므로,
    // 본 벤치마크에선 컨텍스트/백엔드 정리 시 함께 정리된다고 가정.
    inst.finalize_qnn_graph();
    return 0;
}

int main() {
    // =========  환경 설정  =========
    // QNN 라이브러리 경로와 백엔드 이름은 환경변수로 받습니다.
    //   QNN_LIB_DIR: "/vendor/lib/rfsa/adsp/" 혹은 QNN SDK 런타임 so/dll 디렉토리 경로(끝에 슬래시 포함 권장)
    //   QNN_BACKEND: "libQnnHtp.so" (기본), 또는 "libQnnCpu.so" 등
    const char* lib_dir_env = std::getenv("QNN_LIB_DIR");
    const char* backend_env = std::getenv("QNN_BACKEND");

#if !defined(__ANDROID__) && !defined(__linux__)
    const std::string default_backend = backend_env ? backend_env : "QnnHtp.dll";
#else
    const std::string default_backend = backend_env ? backend_env : "libQnnHtp.so";
#endif
    const std::string lib_dir = lib_dir_env ? lib_dir_env : "";
    if (lib_dir.empty()) {
        std::printf("Please set QNN_LIB_DIR to directory containing QNN libs (and ensure libQnnSystem.so present)\n");
    }

    // qnn_instance 생성: (lib_path_dir, backend_name, model_name)
    qnn_instance inst(lib_dir, default_backend, "bench_add");
    if (inst.qnn_init(nullptr) != 0) {
        std::printf("qnn_init failed\n");
        return 1;
    }

    // =========  실험 설정  =========
    const std::vector<std::vector<uint32_t>> shapes = {
        {1, 128},
        {1, 3, 224, 224},
        {64, 64}
    };
    const std::vector<Qnn_DataType_t> dtypes = {
        QNN_DATATYPE_FLOAT_32,
        QNN_DATATYPE_INT_8
    };
    const int alignment_offset = 0;

    // 최대 요소 수
    size_t max_elems = 1;
    for (auto& s : shapes) {
        size_t e = 1;
        for (auto d : s) e *= d;
        max_elems = std::max(max_elems, e);
    }

    // 참조 FP32 출력을 저장해 INT8과 RMSE 비교 시 사용
    std::vector<float> ref_fp32_out(max_elems, 0.0f);

    for (auto& shape : shapes) {
        size_t elems = 1;
        for (auto d : shape) elems *= d;

        std::printf("Shape:");
        for (auto d : shape) std::printf(" %u", d);
        std::printf("\n");

        // 입력 데이터 원본(FP32) 생성
        std::vector<float> src_fp32(elems);
        generate_fp32(0.0f, elems, src_fp32.data());

        for (auto dt : dtypes) {
            const char* dt_name = (dt == QNN_DATATYPE_FLOAT_32 ? "FP32" : "INT8");
            std::printf("  Precision: %s\n", dt_name);

            // dtype별 버퍼 준비(정렬)
            const size_t elem_size = (dt == QNN_DATATYPE_FLOAT_32) ? sizeof(float) : sizeof(int8_t);
            const size_t bytes_needed = elems * elem_size + MAX_ALIGNMENT * 2;

            std::vector<uint8_t> in_buf (bytes_needed);
            std::vector<uint8_t> out_buf(bytes_needed);
            std::vector<uint8_t> z_buf  (bytes_needed); // zero tensor

            void* in_ptr  = align_with_offset(in_buf.data(),  bytes_needed, alignment_offset);
            void* out_ptr = align_with_offset(out_buf.data(), bytes_needed, alignment_offset);
            void* z_ptr   = align_with_offset(z_buf.data(),   bytes_needed, alignment_offset);
            if (!in_ptr || !out_ptr || !z_ptr) {
                std::printf("    Alignment failed\n");
                continue;
            }

            if (dt == QNN_DATATYPE_FLOAT_32) {
                // 입력/제로 초기화
                std::memcpy(in_ptr, src_fp32.data(), elems * sizeof(float));
                std::memset(z_ptr, 0, elems * sizeof(float));
            } else {
                // 간단 캐스팅(실제 양자화 필요 시 scale/zero-point 설정 권장)
                auto* in_i8  = reinterpret_cast<int8_t*>(in_ptr);
                auto* zer_i8 = reinterpret_cast<int8_t*>(z_ptr);
                for (size_t i = 0; i < elems; ++i) in_i8[i] = (int8_t)std::lrintf(src_fp32[i]);
                std::memset(zer_i8, 0, elems * sizeof(int8_t));
            }

            // 그래프 빌드 & 벤치마크
            int64_t min_us = 0;
            double  avg_us = 0.0;
            int rc = build_and_bench_add_graph(inst, shape, dt,
                                               in_ptr, z_ptr, out_ptr,
                                               elems, alignment_offset,
                                               min_us, avg_us);
            if (rc != 0) {
                std::printf("    Graph build/execute failed (rc=%d)\n", rc);
                continue;
            }

            // 결과 수집 & RMSE(옵션)
            if (dt == QNN_DATATYPE_FLOAT_32) {
                std::memcpy(ref_fp32_out.data(), out_ptr, elems * sizeof(float));
            } else {
                // int8 출력을 float로 올려 단순 비교(정량화 파라미터 미설정 상태라 참고용)
                std::vector<float> out_as_fp32(elems);
                const int8_t* out_i8 = reinterpret_cast<const int8_t*>(out_ptr);
                for (size_t i = 0; i < elems; ++i) out_as_fp32[i] = (float)out_i8[i];
                float rmse = compute_rmse(ref_fp32_out.data(), out_as_fp32.data(), elems);
                std::printf("    RMSE (INT8 cast vs FP32): %.6f\n", rmse);
            }

            std::printf("    min latency: %lld us\n", (long long)min_us);
            std::printf("    avg latency: %.2f us\n\n", avg_us);
        }
    }

    inst.qnn_finalize();
    return 0;
}
