/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
void CreateGraphWithEmptyNode(hipGraph_t* graph, hipGraphNode_t* node) {
  HIP_CHECK(hipGraphCreate(graph, 0));
  HIP_CHECK(hipGraphAddEmptyNode(node, *graph, nullptr, 0));
}
}  // namespace

// @asserts: hipGraphNodeGetLocalId - a null node or null output pointer is rejected as an invalid value
HIP_TEST_CASE(Contract_GraphId_HipGraphNodeGetLocalId_NullArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned int node_id = 0;

  CreateGraphWithEmptyNode(&graph, &node);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  REQUIRE(hipGraphNodeGetLocalId(nullptr, &node_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphNodeGetLocalId(node, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphNodeGetToolsId - a null node or null output pointer is rejected as an invalid value
HIP_TEST_CASE(Contract_GraphId_HipGraphNodeGetToolsId_NullArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned long long tools_id = 0;

  CreateGraphWithEmptyNode(&graph, &node);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  REQUIRE(hipGraphNodeGetToolsId(nullptr, &tools_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphNodeGetToolsId(node, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphGetId - a null graph or null output pointer is rejected as an invalid value
HIP_TEST_CASE(Contract_GraphId_HipGraphGetId_NullArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  unsigned int graph_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  REQUIRE(hipGraphGetId(nullptr, &graph_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphGetId(graph, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphExecGetId - a null executable graph or null output pointer is rejected as an invalid value
HIP_TEST_CASE(Contract_GraphId_HipGraphExecGetId_NullArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  unsigned int graph_id = 0;

  CreateGraphWithEmptyNode(&graph, &node);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphInstantiateWithFlags(&graph_exec, graph, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  REQUIRE(hipGraphExecGetId(nullptr, &graph_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphExecGetId(graph_exec, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphNodeGetLocalId - a node's local identifier is stable and differs from another live node's identifier
HIP_TEST_CASE(Contract_GraphId_HipGraphNodeGetLocalId_LiveNodes_AreStableAndDistinct) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t first = nullptr;
  hipGraphNode_t second = nullptr;
  unsigned int first_id = 0;
  unsigned int first_id_repeat = 0;
  unsigned int second_id = 0;

  CreateGraphWithEmptyNode(&graph, &first);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&second, graph, nullptr, 0));

  const hipError_t status = hipGraphNodeGetLocalId(first, &first_id);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Backend does not assign graph node identifiers.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipGraphNodeGetLocalId(first, &first_id_repeat));
  HIP_CHECK(hipGraphNodeGetLocalId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

// @asserts: hipGraphNodeGetToolsId - a node's opaque tools identifier is stable and differs from another live node's identifier
HIP_TEST_CASE(Contract_GraphId_HipGraphNodeGetToolsId_LiveNodes_AreStableAndDistinct) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t first = nullptr;
  hipGraphNode_t second = nullptr;
  unsigned long long first_id = 0;
  unsigned long long first_id_repeat = 0;
  unsigned long long second_id = 0;

  CreateGraphWithEmptyNode(&graph, &first);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&second, graph, nullptr, 0));

  const hipError_t status = hipGraphNodeGetToolsId(first, &first_id);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Backend does not assign graph node identifiers.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipGraphNodeGetToolsId(first, &first_id_repeat));
  HIP_CHECK(hipGraphNodeGetToolsId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

// @asserts: hipGraphGetId - a graph's identifier is stable and differs from another live graph's identifier
HIP_TEST_CASE(Contract_GraphId_HipGraphGetId_LiveGraphs_AreStableAndDistinct) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t first = nullptr;
  hipGraph_t second = nullptr;
  unsigned int first_id = 0;
  unsigned int first_id_repeat = 0;
  unsigned int second_id = 0;

  HIP_CHECK(hipGraphCreate(&first, 0));
  cleanup.Add([first] { (void)hipGraphDestroy(first); });
  HIP_CHECK(hipGraphCreate(&second, 0));
  cleanup.Add([second] { (void)hipGraphDestroy(second); });

  const hipError_t status = hipGraphGetId(first, &first_id);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Backend does not assign graph identifiers.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipGraphGetId(first, &first_id_repeat));
  HIP_CHECK(hipGraphGetId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

// @asserts: hipGraphExecGetId - an executable graph's identifier is stable and differs from another live executable's identifier
HIP_TEST_CASE(Contract_GraphId_HipGraphExecGetId_LiveExecutables_AreStableAndDistinct) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t first = nullptr;
  hipGraphExec_t second = nullptr;
  unsigned int first_id = 0;
  unsigned int first_id_repeat = 0;
  unsigned int second_id = 0;

  CreateGraphWithEmptyNode(&graph, &node);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphInstantiateWithFlags(&first, graph, 0));
  cleanup.Add([first] { (void)hipGraphExecDestroy(first); });
  HIP_CHECK(hipGraphInstantiateWithFlags(&second, graph, 0));
  cleanup.Add([second] { (void)hipGraphExecDestroy(second); });

  const hipError_t status = hipGraphExecGetId(first, &first_id);
  if (status == hipErrorNotSupported) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Backend does not assign executable graph identifiers.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipGraphExecGetId(first, &first_id_repeat));
  HIP_CHECK(hipGraphExecGetId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}
