#!/usr/bin/env bash
#
# Copyright (c) 2024-2025 Jeff Zhou(https://github.com/jeffzhou2000)
#
# 1. build llama.cpp + ggml-hexagon backend on Linux for Android phone equipped with Qualcomm Snapdragon mobile SoC
#    this script will setup local dev envs automatically
#
# 2. verify prebuilt libggmldsp-skel.so on Android phone equipped with Qualcomm Snapdragon mobile SoC(8Elite is recommended)
#
# 3. performance comparison of QNN-CPU,QNN-GPU,QNN-NPU,Hexagon-cDSP,ggml on Android phone equipped with Qualcomm Snapdragon mobile SoC
#
#
set -e

######## part-1: don't modify contents in this part ########

PWD=`pwd`
PROJECT_HOME_PATH=`pwd`
PROJECT_ROOT_PATH=${PROJECT_HOME_PATH}
HOST_CPU_COUNTS=`cat /proc/cpuinfo | grep "processor" | wc | awk '{print int($1)}'`

#running path on Android phone
REMOTE_PATH=/data/local/tmp

#Android NDK can be found at:
#https://developer.android.com/ndk/downloads
ANDROID_PLATFORM=android-34
ANDROID_NDK_VERSION=r28
ANDROID_NDK_NAME=android-ndk-${ANDROID_NDK_VERSION}
ANDROID_NDK_FULLNAME=${ANDROID_NDK_NAME}-linux.zip
ANDROID_NDK=${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_NAME}

#Qualcomm QNN SDK can be found at:
#https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_URL=https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
QNN_SDK_VERSION=2.32.0.250228
QNN_SDK_VERSION=2.33.0.250327
QNN_SDK_VERSION=2.34.0.250424
QNN_SDK_VERSION=2.35.0.250530
QNN_SDK_VERSION=2.36.0.250627
#fully official QNN SDK, will be downloaded automatically via this script
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/qairt/2.34.0.250424/
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/qairt/2.35.0.250530/
QNN_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/qairt/2.36.0.250627/

#fully Qualcomm Hexagon SDK can be found at(fully Hexagon SDK must be obtained with Qualcomm Developer Account):
#https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/6.2.0.1
#the official Qualcomm Hexagon SDK tech docs can be found at:
#https://docs.qualcomm.com/bundle/publicresource/topics/80-77512-1/hexagon-dsp-sdk-collection-landing-page.html?product=1601111740010422
#customized/tailored Hexagon SDK for simplify workflow and can be downloaded via this script
#this highly tailored minimal-hexagon-sdk should comply with Qualcomm's IPR policy.
HEXAGON_SDK_PATH=${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1


#running_params=" -ngl 99 -t 4 -n 256 --no-warmup -fa 1 "
running_params=" -ngl 99 -t 4 -n 256 --no-warmup "

#available prebuilt libs can be found at prebuilts/ggml-dsp
GGMLDSP_RELEASE_DATE=20250531
GGMLDSP_RELEASE_DATE=20250609
GGMLDSP_RELEASE_DATE=20250625
GGMLDSP_RELEASE_DATE=20250627
GGMLDSP_RELEASE_DATE=20250710


######## part-2: contents in this part can be modified ########

PROMPT_STRING="introduce the movie Once Upon a Time in America briefly.\n"

#the following LLM models has verified(works fine) with Hexagon-cDSP backend on a Snapdragon 8Elite based Android phone although inference performance is not good at the moment
#for llama-cli, 6.9 GiB, can be downloadded via this script in function check_prebuilt_models()
TEST_MODEL_NAME=/sdcard/gemma-3n-E4B-it-Q8_0.gguf
#for llama-cli, 8.2 GiB
#TEST_MODEL_NAME=/sdcard/Qwen3-8B-Q8_0.gguf
#for llama-cli, 4.0 GiB
#TEST_MODEL_NAME=/sdcard/Qwen3-4B-Q8_0.gguf
#for llama-cli, 3.9 GiB
#TEST_MODEL_NAME=/sdcard/gemma-3-4b-it-Q8_0.gguf
#for llama-cli, 1.6 GiB, can be downloadded via this script in function check_prebuilt_models()
#TEST_MODEL_NAME=/sdcard/MiniCPM4-0.5B-F32.gguf
#for llama-cli, 4.5 GiB, will be downloadded automatically via this script when running this script at the first time
#TEST_MODEL_NAME=/sdcard/gemma-3n-E2B-it-Q8_0.gguf
#for llama-cli, 1.1 GiB, will be downloaded automatically via this script when running this script at the first time
#this model will be used for compare performance/stability between ggml-opencl, ggml-vulkan, ggml-hexagon on Snapdragon 8Elite
#because the weights of this model is suitable for ggml-opencl,ggml-vulkan,ggml-hexagon on Android phone
TEST_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#for llama-bench, 4.5 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/gemma-3n-E2B-it-Q8_0.gguf
#for llama-bench, 1.12 GiB, will be downloadded automatically via this script when running this script at the first time
GGUF_MODEL_NAME=/sdcard/qwen1_5-1_8b-chat-q4_0.gguf

#supported htp arch version:
#v68 --- Snapdragon 888
#v69 --- Snapdragon 8 Gen1
#v73 --- Snapdragon 8 Gen2
#v75 --- Snapdragon 8 Gen3
#v79 --- Snapdragon 8 Elite

#Qualcomm Snapdragon 8Elite based Android phone is strongly recommended because:
#1. sometimes the same dsp codes can got the best performance on Snapdragon 8Elite based phone.
#2. DSP clock rate on 8Gen3 is slower than DSP clock rate on 8Elite.
#3. 8Elite support for LP-DDR5x memory, up to 5300 MHz; 8Gen3 support for LP-DDR5x memory, up to 4800 MHz.

#modify the following two lines to adapt to test phone
HTP_ARCH_VERSION=V69
HTP_ARCH_VERSION_a=V69

######## part-3: utilities and functions ########

function dump_vars()
{
    echo -e "ANDROID_NDK:          ${ANDROID_NDK}"
    echo -e "QNN_SDK_PATH:         ${QNN_SDK_PATH}"
    echo -e "HEXAGON_SDK_PATH:     ${HEXAGON_SDK_PATH}"
}


function show_pwd()
{
    echo -e "current working path:$(pwd)\n"
}


function check_command_in_host()
{
    set +e
    cmd=$1
    ls /usr/bin/${cmd}
    if [ $? -eq 0 ]; then
        #printf "${cmd} already exist on host machine\n"
        echo ""
    else
        printf "${cmd} not exist on host machine, pls install command line utility ${cmd} firstly and accordingly\n"
        exit 1
    fi
    set -e
}


function check_commands_in_host()
{
    check_command_in_host wget
    check_command_in_host xzcat
}


function check_android_phone()
{
    adb shell ls /bin/ls
    if [ ! $? -eq 0 ]; then
        printf "pls check Android phone is connected properly\n"
        exit 1
    fi
}


function check_and_download_hexagon_sdk()
{
    is_hexagon_llvm_exist=1
    if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/NOTICE.txt ]; then
        echo -e "${TEXT_RED}minimal-hexagon-sdk not exist...${TEXT_RESET}\n"
        is_hexagon_llvm_exist=0
    fi

    if [ ${is_hexagon_llvm_exist} -eq 0 ]; then
        if [ -f ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz ]; then
            echo -e "minimal-hexagon-sdk-6.2.0.1.xz already exist\n"
        else
            echo -e "begin downloading minimal-hexagon-sdk-6.2.0.1.xz \n"
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz https://github.com/jeffzhou2000/toolchain/raw/refs/heads/master/minimal-hexagon-sdk-6.2.0.1.xz
            if [ $? -ne 0 ]; then
                printf "failed to download minimal-hexagon-sdk-6.2.0.1.xz\n"
                exit 1
            fi
        fi

        echo -e "begin decompressing minimal-hexagon-sdk-6.2.0.1.xz \n"
        xzcat ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/minimal-hexagon-sdk-6.2.0.1.xz | tar -C ${PROJECT_ROOT_PATH}/prebuilts/Hexagon_SDK/ -xf -
        if [ $? -ne 0 ]; then
            printf "failed to decompress minimal-hexagon-sdk-6.2.0.1.xz\n"
            exit 1
        fi
        printf "install minimal-hexagon-sdk successfully\n\n"
    fi

    if [ ! -d ${HEXAGON_SDK_PATH} ]; then
        echo -e "HEXAGON_SDK_PATH ${HEXAGON_SDK_PATH} not exist, pls install it accordingly...\n"
        exit 0
    else
        printf "Qualcomm Hexagon SDK already exist:${HEXAGON_SDK_PATH} \n\n"
    fi
}


function check_and_download_qnn_sdk()
{
    is_qnn_sdk_exist=1

    if [ ! -d ${QNN_SDK_PATH} ]; then
        echo -e "QNN_SDK_PATH ${QNN_SDK_PATH} not exist\n"
        is_qnn_sdk_exist=0
    fi

    if [ ! -f ${QNN_SDK_PATH}/NOTICE.txt ]; then
        echo -e "${TEXT_RED}${QNN_SDK_PATH}/NOTICE.txt not exist${TEXT_RESET}\n"
        is_qnn_sdk_exist=0
    fi

    if [ ${is_qnn_sdk_exist} -eq 0 ]; then
        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/v${QNN_SDK_VERSION}.zip ]; then
            echo -e "QNN SDK not exist, download it from ${QNN_SDK_URL}...\n"
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/v${QNN_SDK_VERSION}.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_SDK_VERSION}/v${QNN_SDK_VERSION}.zip
        fi
        if [ $? -ne 0 ]; then
            printf "failed to download Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}/prebuilts/QNN_SDK/
        unzip v${QNN_SDK_VERSION}.zip
        if [ $? -ne 0 ]; then
            printf "failed to decompress Qualcomm QNN SDK to %s \n" "${QNN_SDK_PATH}"
            exit 1
        fi
        printf "Qualcomm QNN SDK saved to ${QNN_SDK_PATH} \n\n"
        cd ${PROJECT_ROOT_PATH}
    else
        printf "Qualcomm QNN SDK already exist:    ${QNN_SDK_PATH} \n\n"
    fi
}


function check_and_download_ndk()
{
    is_android_ndk_exist=1

    if [ ! -d ${ANDROID_NDK} ]; then
        is_android_ndk_exist=0
    fi

    if [ ! -f ${ANDROID_NDK}/build/cmake/android.toolchain.cmake ]; then
        is_android_ndk_exist=0
    fi

    if [ ${is_android_ndk_exist} -eq 0 ]; then

        if [ ! -f ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} ]; then
            wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/prebuilts/${ANDROID_NDK_FULLNAME} https://dl.google.com/android/repository/${ANDROID_NDK_FULLNAME}
        fi

        cd ${PROJECT_ROOT_PATH}/prebuilts/
        unzip ${ANDROID_NDK_FULLNAME}

        if [ $? -ne 0 ]; then
            printf "failed to download Android NDK to %s \n" "${ANDROID_NDK}"
            exit 1
        fi
        cd ${PROJECT_ROOT_PATH}

        printf "Android NDK saved to ${ANDROID_NDK} \n\n"
    else
        printf "Android NDK already exist:         ${ANDROID_NDK} \n\n"
    fi
}


function build_arm64
{
    #not acutually used at the moment, just for AI experts add other AI operators in the future
    #if [ -f ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic ]; then
    #    ${HEXAGON_SDK_PATH}/ipc/fastrpc/qaic/bin/qaic -mdll -o ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels -I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs ${PROJECT_ROOT_PATH}/ggml/src/ggml-hexagon/kernels/ggmlop.idl
    #fi

    cmake -H. -B./out/ggmlhexagon-android -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/ggmlhexagon-android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function build_arm64_debug
{
    cmake -H. -B./out/ggmlhexagon-android -DCMAKE_BUILD_TYPE=Debug -DGGML_OPENMP=OFF -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=latest -DGGML_HEXAGON=ON -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=ON -DQNN_SDK_PATH=${QNN_SDK_PATH} -DHEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} -DHTP_ARCH_VERSION=${HTP_ARCH_VERSION}
    cd out/ggmlhexagon-android
    make -j${HOST_CPU_COUNTS}
    show_pwd

    cd -
}


function remove_temp_dir()
{
    if [ -d out/ggmlhexagon-android ]; then
        echo "remove out/ggmlhexagon-android directory in `pwd`"
        rm -rf out/ggmlhexagon-android
    fi
}


function check_qnn_libs()
{
    set +e

    #reuse the cached qnn libs on Android phone
    adb shell ls ${REMOTE_PATH}/libQnnCpu.so
    adb shell ls ${REMOTE_PATH}/libQnnGpu.so
    adb shell ls ${REMOTE_PATH}/libQnnHtp.so
    if [ $? -eq 0 ]; then
        printf "QNN runtime libs already exist on Android phone\n"
    else
        printf "QNN runtime libs not exist on Android phone\n"
        update_qnn_libs
    fi
    update_qnn_cfg

    set -e
}


function update_qnn_libs()
{
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so              ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so                 ${REMOTE_PATH}/

    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so                 ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp${HTP_ARCH_VERSION_a}Stub.so          ${REMOTE_PATH}/
    adb push ${QNN_SDK_PATH}/lib/hexagon-${HTP_ARCH_VERSION}/unsigned/libQnnHtp${HTP_ARCH_VERSION_a}Skel.so     ${REMOTE_PATH}/
}


function update_qnn_cfg()
{
    adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/
}


function build_ggml_hexagon()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64
}


function build_ggml_hexagon_debug()
{
    show_pwd
    check_and_download_ndk
    check_and_download_qnn_sdk
    check_and_download_hexagon_sdk
    dump_vars
    remove_temp_dir
    build_arm64_debug
}


function prepare_ggmldsp()
{
    adb push ./scripts/ggml-hexagon-for-binary-lib.cfg ${REMOTE_PATH}/ggml-hexagon.cfg
    echo "adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so"
case "$HTP_ARCH_VERSION" in
    v69)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v73)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v75)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    v79)
        adb push ${PROJECT_ROOT_PATH}/prebuilts/ggml-dsp/${GGMLDSP_RELEASE_DATE}/libggmldsp-skel${HTP_ARCH_VERSION}.so ${REMOTE_PATH}/libggmldsp-skel.so
    ;;
    *)
        show_usage
        exit 1
    ;;
esac
}


function check_and_download_model()
{
    set +e

    model_name=$1
    model_url=$2

    adb shell ls /sdcard/${model_name}
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model ${model_name} already exist on Android phone\n"
    else
        printf "the prebuild LLM model ${model_name} not exist on Android phone\n"
        wget --no-config --quiet --show-progress -O ${PROJECT_ROOT_PATH}/models/${model_name} ${model_url}
        adb push ${PROJECT_ROOT_PATH}/models/${model_name} /sdcard/
    fi

    set -e
}


function check_prebuilt_models()
{
    set +e

    adb shell ls /sdcard/t5-very-small-random-F32.gguf
    if [ $? -eq 0 ]; then
        printf "the prebuild LLM model t5-very-small-random-F32.gguf already exist on Android phone\n"
    else
        printf "the prebuild LLM model t5-very-small-random-F32.gguf not exist on Android phone\n"
        adb push ${PROJECT_ROOT_PATH}/models/t5-very-small-random-F32.gguf /sdcard/
    fi

    #1.12 GiB
    check_and_download_model qwen1_5-1_8b-chat-q4_0.gguf https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/resolve/main/qwen1_5-1_8b-chat-q4_0.gguf
    #6.9 GiB
    #check_and_download_model gemma-3n-E4B-it-Q8_0.gguf https://huggingface.co/ggml-org/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q8_0.gguf
    #4.5 GiB
    check_and_download_model gemma-3n-E2B-it-Q8_0.gguf https://huggingface.co/ggml-org/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q8_0.gguf

    set -e
}


function prepare_run_on_phone()
{
    if [ $# != 1 ]; then
        print "invalid param"
        return
    fi
    program=$1

    check_qnn_libs

    check_prebuilt_models

    if [ -f ./out/ggmlhexagon-android/bin/libggml-cpu.so ]; then
        adb push ./out/ggmlhexagon-android/bin/*.so ${REMOTE_PATH}/
    fi
    adb push ./out/ggmlhexagon-android/bin/${program} ${REMOTE_PATH}/

    #for troubleshooting issues in upstream llama.cpp project
    adb shell ls -l ${REMOTE_PATH}/libggml-*.so

    #for verify prebuilt binary library(after 06/2025) on Hexagon cDSP
    #comment this line when build library on Hexagon cDSP from the reference/self-develop source codes in this project
    prepare_ggmldsp

    #un-comment this line when build library on Hexagon cDSP from the reference/self-develop source codes in this project
    #adb push ./scripts/ggml-hexagon.cfg ${REMOTE_PATH}/ggml-hexagon.cfg

    adb shell chmod +x ${REMOTE_PATH}/${program}
}

function run_llamacli()
{
    prepare_run_on_phone llama-cli

    echo "${REMOTE_PATH}/llama-cli ${running_params} -mg ${hexagon_backend} -no-cnv -m ${TEST_MODEL_NAME} -p \"${PROMPT_STRING}\""
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-cli ${running_params} -mg ${hexagon_backend} -no-cnv -m ${TEST_MODEL_NAME} -p \"${PROMPT_STRING}\""

}


function run_llamabench()
{
    prepare_run_on_phone llama-bench

    echo "adb shell \"cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -mg ${hexagon_backend} -m ${GGUF_MODEL_NAME}\""
    echo "${REMOTE_PATH}/llama-bench ${running_params} -mg ${hexagon_backend} -m ${GGUF_MODEL_NAME}"

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/llama-bench ${running_params} -mg ${hexagon_backend} -m ${GGUF_MODEL_NAME}"

}


function run_threadsafety()
{
    prepare_run_on_phone test-thread-safety

    echo "${REMOTE_PATH}/test-thread-safety -np 2 -mg ${hexagon_backend} -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/test-thread-safety -np 1 -mg ${hexagon_backend} -m ${GGUF_MODEL_NAME} -p \"hello,world\" -n 256 -ngl 99 "

}


function run_test-ops()
{
    prepare_run_on_phone ggmlhexagon-testops

    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-testops test"

}


function check_hexagon_backend
{
    if [[ ${hexagon_backend} != 0 ]] && [[ ${hexagon_backend} != 1 ]] && [[ ${hexagon_backend} != 2 ]] && [[ ${hexagon_backend} != 3 ]] && [[ ${hexagon_backend} != 4 ]] ; then
        printf "invalid hexagon backend\n"
        printf "valid hexagon backend: 0(QNN_CPU), 1(QNN_GPU), 2(QNN_NPU), 3(cDSP), 4(ggml)\n"
        exit 1
    fi
}


function check_mulmat_algotype
{
    printf "mulmat_algotype ${mulmat_algotype} \n"
    if [[ ${mulmat_algotype} != 0 ]] && [[ ${mulmat_algotype} != 1 ]] && [[ ${mulmat_algotype} != 2 ]] && [[ ${mulmat_algotype} != 3 ]] && [[ ${mulmat_algotype} != 4 ]] && [[ ${mulmat_algotype} != 5 ]] && [[ ${mulmat_algotype} != 6 ]] && [[ ${mulmat_algotype} != 32 ]] && [[ ${mulmat_algotype} != 33 ]]; then
        printf "invalid mulmat algotype\n"
        printf "valid mulmat algotype: 0, 1, 2, 3, 4, 5, 6, 32, 33 \n"
        exit 1
    fi
}


function run_test-op()
{
    prepare_run_on_phone ggmlhexagon-testops

    check_mulmat_algotype

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-testops test -o ${opname} -a ${mulmat_algotype} -i ${hexagon_backend}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-testops test -o ${opname} -a ${mulmat_algotype} -i ${hexagon_backend}"

}


function run_perf-op()
{
    prepare_run_on_phone ggmlhexagon-testops

    check_mulmat_algotype

    echo "adb shell cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-testops perf -o ${opname} -a ${mulmat_algotype} -i ${hexagon_backend}"

    echo "\n"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-testops perf -o ${opname} -a ${mulmat_algotype} -i ${hexagon_backend}"

}


function run_benchmark()
{
    prepare_run_on_phone ggmlhexagon-benchmark

    check_mulmat_algotype

    echo "${REMOTE_PATH}/ggmlhexagon-benchmark -t ${opname} -b ${hexagon_backend} -m ${row} -n ${col} -a ${mulmat_algotype}"
    adb shell "cd ${REMOTE_PATH} \
               && export LD_LIBRARY_PATH=${REMOTE_PATH} \
               && ${REMOTE_PATH}/ggmlhexagon-benchmark -t ${opname} -b ${hexagon_backend} -m ${row} -n ${col} -a ${mulmat_algotype}"

}


function print_oplist()
{
oplist="DUP
    ADD
    ADD1
    ACC
    SUB
    MUL
    DIV
    SQR
    SQRT
    LOG
    SIN
    COS
    SUM
    SUM_ROWS
    MEAN
    ARGMAX
    COUNT_EQUAL
    REPEAT
    REPEAT_BACK
    CONCAT
    SILU_BACK
    NORM
    RMS_NORM
    RMS_NORM_BACK
    GROUP_NORM

    MUL_MAT
    MUL_MAT_ID
    OUT_PROD

    SCALE
    SET
    CPY
    CONT
    RESHAPE
    VIEW
    PERMUTE
    TRANSPOSE
    GET_ROWS
    GET_ROWS_BACK
    DIAG
    DIAG_MASK_INF
    DIAG_MASK_ZERO
    SOFT_MAX
    SOFT_MAX_BACK
    ROPE
    ROPE_BACK
    CLAMP
    CONV_TRANSPOSE_1D
    IM2COL
    IM2COL_BACK
    CONV_TRANSPOSE_2D
    POOL_1D
    POOL_2D
    POOL_2D_BACK
    UPSCALE
    PAD
    PAD_REFLECT_1D
    ARANGE
    TIMESTEP_EMBEDDING
    ARGSORT
    LEAKY_RELU

    FLASH_ATTN_EXT
    FLASH_ATTN_BACK
    SSM_CONV
    SSM_SCAN
    WIN_PART
    WIN_UNPART
    GET_REL_POS
    ADD_REL_POS
    RWKV_WKV6
    GATED_LINEAR_ATTN"

echo "opname list: "
echo ${oplist}
}


function show_usage()
{
    echo -e "\n\n\n"
    echo "Usage:"
    echo "  $0 help"
    echo "  $0 print_oplist"
    echo "  $0 build"
    echo "  $0 build_debug (enable debug log for developers on ARM-AP side and cDSP side)"
    echo "  $0 updateqnnlib"
    echo "  $0 run_testops"
    echo "  $0 run_testop     ADD/MUL_MAT                                                    (verify accuracy    of ADD/MUL_MAT)"
    echo "  $0 run_perfop     ADD/MUL_MAT                                                    (verify performance of ADD/MUL_MAT)"
    echo "  $0 run_llamacli                 0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml)"
    echo "  $0 run_llamabench               0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml)"
    echo "  $0 run_threadsafety             0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml)"
    echo "  $0 run_perfop     MUL_MAT       0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml) (verify performance of MUL_MAT)"
    echo "  $0 run_testop     MUL_MAT       0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml) (verify accuracy    of MUL_MAT)"
    echo "  $0 run_benchmark  ADD/MUL_MAT   0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml) (verify performance of ADD/MUL_MAT)"
    echo "  $0 run_benchmark  ADD/MUL_MAT   0(QNN_CPU)/1(QNN_GPU)/2(QNN_NPU)/3(cdsp)/4(ggml) 256/512/1024/2048/4096 256/512/1024/2048/4096"
    echo "  $0 run_benchmark  MUL_MAT       3(cdsp)   mulmat_algotype(0,1,2,3,4,5,6,32,33)   (verify performance of MUL_MAT on cDSP)"
    echo "  $0 run_perfop     MUL_MAT       3(cdsp)   mulmat_algotype(0,1,2,3,4,5,6,32,33)   (verify performance of MUL_MAT on cDSP)"
    echo "  $0 run_testop     MUL_MAT       3(cdsp)   mulmat_algotype(0,1,2,3,4,5,6,32,33)   (verify accuracy    of MUL_MAT on cDSP)"
    echo "  $0 run_testop     ADD                                                            (verify accuracy    of ADD     on cDSP)"

    echo -e "\n\n\n"
}


######## part-4: entry point  ########

show_pwd

# check_commands_in_host
# check_android_phone
# check_and_download_ndk
# check_and_download_qnn_sdk
# check_and_download_hexagon_sdk
# check_prebuilt_models

if [ $# == 0 ]; then
    show_usage
    exit 1
elif [ $# == 1 ]; then
    if [ "$1" == "-h" ]; then
        show_usage
        exit 1
    elif [ "$1" == "help" ]; then
        show_usage
        exit 1
    elif [ "$1" == "print_oplist" ]; then
        print_oplist
        exit 1
    elif [ "$1" == "build" ]; then
        build_ggml_hexagon
        exit 0
    elif [ "$1" == "build_debug" ]; then
        build_ggml_hexagon_debug
        exit 0
    elif [ "$1" == "run_testops" ]; then
        run_test-ops
        exit 0
    elif [ "$1" == "updateqnnlib" ]; then
        update_qnn_libs
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 2 ]; then
#TODO: check opname in oplist
#opname can be found via print_oplist:

    if [ "$1" == "run_testop" ]; then
        opname=$2
        mulmat_algotype=32
        hexagon_backend=3
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=$2
        mulmat_algotype=32
        hexagon_backend=3
        check_hexagon_backend
        run_perf-op
        exit 0
    elif [ "$1" == "run_llamacli" ]; then
        hexagon_backend=$2
        check_hexagon_backend
        run_llamacli
        exit 0
    elif [ "$1" == "run_llamabench" ]; then
        hexagon_backend=$2
        check_hexagon_backend
        run_llamabench
        exit 0
    elif [ "$1" == "run_threadsafety" ]; then
        hexagon_backend=$2
        check_hexagon_backend
        run_threadsafety
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 3 ]; then
    if [ "$1" == "run_benchmark" ]; then
        opname=$2
        hexagon_backend=$3
        row=4096
        col=4096
        mulmat_algotype=32
        check_hexagon_backend
        run_benchmark
        exit 0
    elif [ "$1" == "run_testop" ]; then
        opname=MUL_MAT
        mulmat_algotype=32
        hexagon_backend=$3
        check_hexagon_backend
        run_test-op
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=MUL_MAT
        mulmat_algotype=32
        hexagon_backend=$3
        run_perf-op
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 4 ]; then
    if [ "$1" == "run_benchmark" ]; then
        opname=MUL_MAT
        #cDSP
        hexagon_backend=3
        row=4096
        col=4096
        mulmat_algotype=$4
        check_mulmat_algotype
        run_benchmark
        exit 0
    elif [ "$1" == "run_perfop" ]; then
        opname=MUL_MAT
        hexagon_backend=3
        mulmat_algotype=$4
        check_mulmat_algotype
        run_perf-op
        exit 0
    elif [ "$1" == "run_testop" ]; then
        opname=MUL_MAT
        hexagon_backend=3
        mulmat_algotype=$4
        check_mulmat_algotype
        run_test-op
        exit 0
    else
        show_usage
        exit 1
    fi
elif [ $# == 5 ]; then
    if [ "$1" == "run_benchmark" ]; then
        opname=$2
        hexagon_backend=$3
        row=$4
        col=$5
        mulmat_algotype=32
        check_hexagon_backend
        run_benchmark
        exit 0
    else
        show_usage
        exit 1
    fi
else
    show_usage
    exit 1
fi
