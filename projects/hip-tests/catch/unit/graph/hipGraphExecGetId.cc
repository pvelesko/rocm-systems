/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

// BACKEND-DIFF: CUDA introduced graph identifier queries in 13.1. Older NVIDIA toolkits
// cannot exercise these tests.
#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13010)

/**
 * @addtogroup hipGraphExecGetId hipGraphExecGetId
 * @{
 * @ingroup GraphTest
 * `hipGraphExecGetId(hipGraphExec_t hGraphExec, unsigned int *graphID)` -
 * Returns the id of a given graph exec
 */

static __global__ void emptyKernel() {}

/**
 * Test Description
 * ------------------------
 *    - Verify that multiple instantiations produce unique exec IDs:
 *        -# Instantiate the same graph twice
 *        -# Verify both exec IDs are different from each other and from the graph ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_UniquePerInstantiation) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t emptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));

  unsigned int graphId = 0;
  HIP_CHECK(hipGraphGetId(graph, &graphId));

  hipGraphExec_t exec1, exec2;
  HIP_CHECK(hipGraphInstantiate(&exec1, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&exec2, graph, nullptr, nullptr, 0));

  unsigned int execId1 = 0, execId2 = 0;
  HIP_CHECK(hipGraphExecGetId(exec1, &execId1));
  HIP_CHECK(hipGraphExecGetId(exec2, &execId2));

  REQUIRE(graphId < execId1);
  REQUIRE(execId1 < execId2);

  HIP_CHECK(hipGraphExecDestroy(exec1));
  HIP_CHECK(hipGraphExecDestroy(exec2));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify graph and executable graph IDs share one creation-order counter:
 *        -# Create a graph, instantiate it, then repeat with a second graph
 *        -# Verify all four IDs are strictly increasing in creation order
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_SharesCounterWithGraphId) {
  hipGraph_t graph1;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  hipGraphNode_t node1;
  HIP_CHECK(hipGraphAddEmptyNode(&node1, graph1, nullptr, 0));
  unsigned int graphId1;
  HIP_CHECK(hipGraphGetId(graph1, &graphId1));

  hipGraphExec_t exec1;
  HIP_CHECK(hipGraphInstantiate(&exec1, graph1, nullptr, nullptr, 0));
  unsigned int execId1;
  HIP_CHECK(hipGraphExecGetId(exec1, &execId1));

  hipGraph_t graph2;
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  hipGraphNode_t node2;
  HIP_CHECK(hipGraphAddEmptyNode(&node2, graph2, nullptr, 0));
  unsigned int graphId2;
  HIP_CHECK(hipGraphGetId(graph2, &graphId2));

  hipGraphExec_t exec2;
  HIP_CHECK(hipGraphInstantiate(&exec2, graph2, nullptr, nullptr, 0));
  unsigned int execId2;
  HIP_CHECK(hipGraphExecGetId(exec2, &execId2));

  REQUIRE(graphId1 < execId1);
  REQUIRE(execId1 < graphId2);
  REQUIRE(graphId2 < execId2);

  HIP_CHECK(hipGraphExecDestroy(exec1));
  HIP_CHECK(hipGraphExecDestroy(exec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphExecGetId works with a graph from stream capture:
 *        -# Capture a stream, instantiate, verify exec ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_FromCapturedStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  emptyKernel<<<1, 1, 0, stream>>>();
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  unsigned int execId = 0;
  HIP_CHECK(hipGraphExecGetId(graphExec, &execId));
  REQUIRE(execId >= 1);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Verify exec IDs from a cloned graph are different:
 *        -# Create and clone a graph
 *        -# Instantiate both
 *        -# Verify exec IDs are different
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_FromClonedGraph) {
  hipGraph_t graph, clonedGraph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t emptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));

  HIP_CHECK(hipGraphClone(&clonedGraph, graph));

  hipGraphExec_t origExec, clonedExec;
  HIP_CHECK(hipGraphInstantiate(&origExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&clonedExec, clonedGraph, nullptr, nullptr, 0));

  unsigned int origExecId, clonedExecId;
  HIP_CHECK(hipGraphExecGetId(origExec, &origExecId));
  HIP_CHECK(hipGraphExecGetId(clonedExec, &clonedExecId));

  REQUIRE(origExecId != clonedExecId);

  HIP_CHECK(hipGraphExecDestroy(origExec));
  HIP_CHECK(hipGraphExecDestroy(clonedExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify exec ID is stable after hipGraphExecUpdate:
 *        -# Instantiate a graph, get exec ID
 *        -# Create a second compatible graph and update the exec with it
 *        -# Verify the exec ID is unchanged
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_StableAfterExecUpdate) {
  hipGraph_t graph1;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  hipGraphNode_t n1;
  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(emptyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.sharedMemBytes = 0;
  kp.kernelParams = nullptr;
  kp.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&n1, graph1, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph1, nullptr, nullptr, 0));

  unsigned int execIdBefore;
  HIP_CHECK(hipGraphExecGetId(exec, &execIdBefore));

  // Create a compatible graph and update the exec
  hipGraph_t graph2;
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  hipGraphNode_t n2;
  HIP_CHECK(hipGraphAddKernelNode(&n2, graph2, nullptr, 0, &kp));

  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult;
  HIP_CHECK(hipGraphExecUpdate(exec, graph2, &errorNode, &updateResult));

  unsigned int execIdAfter;
  HIP_CHECK(hipGraphExecGetId(exec, &execIdAfter));

  REQUIRE(execIdBefore == execIdAfter);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify exec ID is stable after hipGraphLaunch:
 *        -# Instantiate a graph, get exec ID
 *        -# Launch and synchronize
 *        -# Verify the exec ID is unchanged
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_StableAfterLaunch) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node;
  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(emptyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.sharedMemBytes = 0;
  kp.kernelParams = nullptr;
  kp.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  unsigned int idBefore;
  HIP_CHECK(hipGraphExecGetId(exec, &idBefore));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  unsigned int idAfter;
  HIP_CHECK(hipGraphExecGetId(exec, &idAfter));

  REQUIRE(idBefore == idAfter);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify exec IDs are not reused after destroy and reinstantiate:
 *        -# Instantiate a graph, get exec ID, destroy exec
 *        -# Instantiate the same graph again
 *        -# Verify the new exec has a different ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_NoReuseAfterDestroy) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  hipGraphExec_t exec1;
  HIP_CHECK(hipGraphInstantiate(&exec1, graph, nullptr, nullptr, 0));
  unsigned int id1;
  HIP_CHECK(hipGraphExecGetId(exec1, &id1));
  HIP_CHECK(hipGraphExecDestroy(exec1));

  hipGraphExec_t exec2;
  HIP_CHECK(hipGraphInstantiate(&exec2, graph, nullptr, nullptr, 0));
  unsigned int id2;
  HIP_CHECK(hipGraphExecGetId(exec2, &id2));
  HIP_CHECK(hipGraphExecDestroy(exec2));

  REQUIRE(id1 != id2);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
