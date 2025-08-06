// test-npu.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <vector>
#include <tuple>
#include <functional>
#include "ggml.h"
#include "ggml-hexagon.h"  // qnn_instance, Qnn_* API

#define MAX_ALIGNMENT 64

// RAII 래퍼
struct GraphGuard {
    qnn_instance &inst;
    Qnn_GraphHandle_t graph;
    GraphGuard(qnn_instance &i): inst(i), graph(Qnn_GraphCreate(i.handle())) {}
    ~GraphGuard() { Qnn_GraphDestroy(graph); }
    operator Qnn_GraphHandle_t() const { return graph; }
};
struct TensorGuard {
    Qnn_Tensor_t *t;
    TensorGuard(Qnn_Tensor_t *t_): t(t_) {}
    ~TensorGuard() { Qnn_TensorDestroy(t); }
    operator Qnn_Tensor_t*() const { return t; }
};

static int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        high_resolution_clock::now().time_since_epoch()
    ).count();
}

static void benchmark_fn(const std::string &tag,
                         const std::function<void()> &fn,
                         int warmup, int iterations,
                         int64_t &out_min, double &out_avg) {
    // 웜업
    for (int i = 0; i < warmup; i++) fn();
    // 벤치
    int64_t t_min = INT64_MAX, t_sum = 0;
    for (int i = 0; i < iterations; i++) {
        auto t0 = now_us();
        fn();
        auto dt = now_us() - t0;
        t_min = std::min(t_min, dt);
        t_sum += dt;
    }
    out_min = t_min;
    out_avg = t_sum / double(iterations);
    printf("[%s] min %lld µs, avg %.2f µs\n",
           tag.c_str(), (long long)t_min, out_avg);
}

// ggml_tensor → QNN 텐서
static Qnn_Tensor_t * create_qnn_tensor(
    qnn_instance &inst,
    Qnn_GraphHandle_t graph,
    ggml_tensor *tensor,
    Qnn_TensorType_t default_type)
{
    uint32_t dims[4] = {
        uint32_t(tensor->ne[0]),
        uint32_t(tensor->ne[1]),
        uint32_t(tensor->ne[2]),
        uint32_t(tensor->ne[3])
    };
    Qnn_DataType_t dt = QNN_DATATYPE_FLOAT_32;
    Qnn_TensorType_t tt = default_type;
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT)  tt = QNN_TENSOR_TYPE_APP_WRITE;
    if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) tt = QNN_TENSOR_TYPE_APP_READ;
    Qnn_Tensor_t *q = Qnn_GraphCreateTensor(
        graph, dt, tt, 4, dims, tensor->data
    );
    if (!q) {
        fprintf(stderr, "ERROR: GraphCreateTensor failed\n");
        std::exit(1);
    }
    return q;
}

int main(int argc, char **argv) {
    // 기본 파라미터
    std::vector<std::tuple<int,int,int>> shapes = {
        {32,32,32}, {64,64,64}, {128,128,128},
        {64,128,64}, {128,64,256}, {256,256,64}
    };
    std::vector<ggml_type> precisions = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0
    };
    std::vector<int> offsets = {0};  // 얼라인먼트 오프셋
    int warmup = 5, iterations = 10;
    std::string mode = "qnn";  // or "cdsp"
    std::string csv_path = "results.csv";

    // 커맨드라인 옵션
    struct option long_opts[] = {
        {"shape",      required_argument, 0, 's'},
        {"precision",  required_argument, 0, 'p'},
        {"offset",     required_argument, 0, 'o'},
        {"warmup",     required_argument, 0, 'w'},
        {"iter",       required_argument, 0, 'i'},
        {"mode",       required_argument, 0, 'm'},
        {"output",     required_argument, 0, 'f'},
        {0,0,0,0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "s:p:o:w:i:m:f:", long_opts, nullptr)) != -1) {
        switch (opt) {
        case 's':
            // 예: --shape 128x64x256
            {
                int M,K,N;
                if (sscanf(optarg, "%dx%dx%d", &M,&K,&N)==3)
                    shapes.push_back({M,K,N});
            }
            break;
        case 'p':
            // 예: --precision f16/q4_0
            if (!strcmp(optarg,"f32")) precisions.push_back(GGML_TYPE_F32);
            if (!strcmp(optarg,"f16")) precisions.push_back(GGML_TYPE_F16);
            if (!strcmp(optarg,"q4_0"))precisions.push_back(GGML_TYPE_Q4_0);
            break;
        case 'o':
            offsets.clear();
            offsets.push_back(atoi(optarg));
            break;
        case 'w': warmup = atoi(optarg); break;
        case 'i': iterations = atoi(optarg); break;
        case 'm': mode = optarg; break;
        case 'f': csv_path = optarg; break;
        default: break;
        }
    }

    // CSV 파일 열기
    FILE *csv = fopen(csv_path.c_str(), "w");
    fprintf(csv, "mode,precision,M,K,N,offset,min_us,avg_us\n");

    // 최대 버퍼 사이즈 계산
    size_t max_elems = 0;
    for (auto &t: shapes) {
        auto [M,K,N] = t;
        max_elems = std::max({max_elems, size_t(M)*K, size_t(K)*N, size_t(M)*N});
    }
    std::vector<uint8_t> bufA(max_elems*4 + MAX_ALIGNMENT);
    std::vector<uint8_t> bufB(max_elems*4 + MAX_ALIGNMENT);

    // 메모리 얼라인
    auto align = [&](uint8_t *base, int off){
        size_t space = bufA.size();
        void *p = std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, (void*)base, space);
        return (float*)((char*)p + off);
    };
    float *A = align(bufA.data(), offsets[0]);
    float *B = align(bufB.data(), offsets[0]);
    for (size_t i = 0; i < max_elems; i++) {
        A[i] = 0.1f + 2.0f * cosf(i + 0.0f);
        B[i] = 0.1f + 2.0f * cosf(i + 1.0f);
    }

    // GGML + Hexagon 초기화
    size_t mem_size = 64ull * 1024 * 1024;
    void *mem_buffer = malloc(mem_size);
    ggml_init_params ip{ mem_size, mem_buffer, false };
    ggml_context *ctx = ggml_init(ip);
    qnn_instance &inst = qnn_instance::get();
    ggml_backend_hexagon_device_init_backend();  // 확실히 초기화

    // 벤치마크 루프
    for (auto type: precisions) {
        ggml_quantize_init(type);
        const char *ptype = ggml_type_name(type);
        for (auto &t: shapes) {
            auto [M,K,N] = t;
            // 메타 텐서
            ggml_tensor *tA = ggml_new_tensor_2d(ctx, type, M, K);
            ggml_tensor *tB = ggml_new_tensor_2d(ctx, type, K, N);
            ggml_tensor *tC = ggml_new_tensor_2d(ctx, type, M, N);
            tA->data = A; tB->data = B; tC->data = nullptr;

            // 그래프 생성
            GraphGuard graph(inst);
            // 입력/가중치/출력 QNN 텐서
            TensorGuard qA(create_qnn_tensor(inst, graph, tA, QNN_TENSOR_TYPE_APP_WRITE));
            TensorGuard qB(create_qnn_tensor(inst, graph, tB, QNN_TENSOR_TYPE_CONST));
            TensorGuard qC(create_qnn_tensor(inst, graph, tC, QNN_TENSOR_TYPE_APP_READ));

            // MatMul 노드
            Qnn_ErrorHandle_t err = Qnn_GraphAddMatMulNode(
                graph, qC, qA, qB, 0
            );
            if (err != QNN_SUCCESS) {
                fprintf(stderr, "addMatMulNode error %d\n", err);
                return 1;
            }
            // 그래프 파이널라이즈
            err = inst.raw_interface.graphFinalize(graph, nullptr, nullptr);
            if (err != QNN_SUCCESS) {
                fprintf(stderr, "graphFinalize error %d\n", err);
                return 2;
            }

            // 벤치
            std::string tag = mode + "_" + ptype + "_" +
                              std::to_string(M) + "x" +
                              std::to_string(K) + "x" +
                              std::to_string(N);
            int64_t tmin; double tavg;
            benchmark_fn(tag, [&]{
                if (mode == "qnn") {
                    Qnn_GraphExecute(graph);
                } else {
                    // cDSP 스켈레톤 호출 예시
                    ggmlhexagon_op_func_t func = ggmlhexagon_get_op_func(M,K,N);
                    func((uint8_t*)tA->data, (uint8_t*)tB->data, (uint8_t**)&tC->data);
                }
            }, warmup, iterations, tmin, tavg);

            // CSV 기록
            fprintf(csv, "%s,%s,%d,%d,%d,%d,%lld,%.2f\n",
                    mode.c_str(), ptype,
                    M, K, N, offsets[0],
                    (long long)tmin, tavg);
        }
    }

    // 정리
    fclose(csv);
    ggml_free(ctx);
    free(mem_buffer);
    return 0;
}
