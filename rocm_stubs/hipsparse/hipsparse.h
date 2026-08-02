#pragma once
// Stub for paroquant NixOS build — paroquant kernel does not use hipsparse
typedef void* hipsparseHandle_t;
typedef int   hipsparseStatus_t;
typedef void* hipsparseSpMatDescr_t;
typedef void* hipsparseDnVecDescr_t;
typedef void* hipsparseDnMatDescr_t;
#ifdef __cplusplus
extern "C" {
#endif
const char* hipsparseGetErrorString(hipsparseStatus_t status);
#ifdef __cplusplus
}
#endif
