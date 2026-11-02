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
#include <set>
#include <vector>

// BACKEND-DIFF: CUDA introduced graph identifier queries in 13.1. Older NVIDIA toolkits
// cannot exercise these tests.
#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13010)

/**
 * @addtogroup hipGraphNodeGetToolsId hipGraphNodeGetToolsId
 * @{
 * @ingroup GraphTest
 * `hipGraphNodeGetToolsId(hipGraphNode_t hNode, unsigned long long *toolsNodeId)` -
 * Returns an id used by tools to identify a given node
 */

static __global__ void emptyKernel() {}

/**
 * Test Description
 * ------------------------
 *    - Verify that tools IDs are unique within a graph:
 *        -# Create a graph with multiple empty nodes
 *        -# Verify all their tools IDs are distinct
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_UniqueWithinGraph) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  constexpr int numNodes = 5;
  hipGraphNode_t nodes[numNodes];
  unsigned long long ids[numNodes];

  for (int i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphAddEmptyNode(&nodes[i], graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetToolsId(nodes[i], &ids[i]));
  }

  std::set<unsigned long long> idSet(ids, ids + numNodes);
  REQUIRE(idSet.size() == numNodes);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify tools IDs are unique for different node types:
 *        -# Create a graph with empty and kernel nodes
 *        -# Verify their tools IDs are distinct
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_DifferentNodeTypes) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Empty node
  hipGraphNode_t emptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));

  // Kernel node
  hipGraphNode_t kernelNode;
  hipKernelNodeParams kernelParams = {};
  void* kernelFunc = reinterpret_cast<void*>(emptyKernel);
  kernelParams.func = kernelFunc;
  kernelParams.gridDim = dim3(1);
  kernelParams.blockDim = dim3(1);
  kernelParams.sharedMemBytes = 0;
  kernelParams.kernelParams = nullptr;
  kernelParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nullptr, 0,
                                   &kernelParams));

  unsigned long long emptyToolsId, kernelToolsId;
  HIP_CHECK(hipGraphNodeGetToolsId(emptyNode, &emptyToolsId));
  HIP_CHECK(hipGraphNodeGetToolsId(kernelNode, &kernelToolsId));

  REQUIRE(emptyToolsId != kernelToolsId);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify tools IDs are unique in a graph from stream capture:
 *        -# Capture a stream with multiple kernel launches
 *        -# Extract nodes and verify all tools IDs are unique
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_CapturedStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  emptyKernel<<<1, 1, 0, stream>>>();
  emptyKernel<<<1, 1, 0, stream>>>();

  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes >= 2);

  std::vector<hipGraphNode_t> nodes(numNodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &numNodes));

  std::set<unsigned long long> idSet;
  for (size_t i = 0; i < numNodes; i++) {
    unsigned long long toolsId;
    HIP_CHECK(hipGraphNodeGetToolsId(nodes[i], &toolsId));
    idSet.insert(toolsId);
  }

  REQUIRE(idSet.size() == numNodes);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphNodeGetToolsId works on a host node:
 *        -# Create a graph with a host node
 *        -# Verify hipGraphNodeGetToolsId returns successfully
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
static void dummyHostFn(void*) {}

HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_HostNode) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipHostNodeParams hostParams = {};
  hostParams.fn = dummyHostFn;
  hostParams.userData = nullptr;

  hipGraphNode_t hostNode;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  unsigned int graphId;
  HIP_CHECK(hipGraphGetId(graph, &graphId));
  unsigned int localId;
  HIP_CHECK(hipGraphNodeGetLocalId(hostNode, &localId));
  unsigned long long toolsId;
  HIP_CHECK(hipGraphNodeGetToolsId(hostNode, &toolsId));

  REQUIRE(graphId >= 1);
  REQUIRE(localId == 0);
  REQUIRE(toolsId == (static_cast<unsigned long long>(graphId) << 32));

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify that tools ID encodes graph ID and local node ID:
 *        -# Create a graph with multiple nodes
 *        -# Verify local IDs increment from zero and toolsId == (graphId << 32) | localId
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_EncodesGraphIdAndLocalId) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  unsigned int graphId;
  HIP_CHECK(hipGraphGetId(graph, &graphId));

  constexpr int numNodes = 3;
  for (int i = 0; i < numNodes; i++) {
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

    unsigned int localId;
    unsigned long long toolsId;
    HIP_CHECK(hipGraphNodeGetLocalId(node, &localId));
    HIP_CHECK(hipGraphNodeGetToolsId(node, &toolsId));

    REQUIRE(localId == static_cast<unsigned int>(i));
    unsigned long long expected =
        (static_cast<unsigned long long>(graphId) << 32) | localId;
    REQUIRE(toolsId == expected);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify tools IDs are globally unique across multiple graphs:
 *        -# Create two graphs with multiple nodes each
 *        -# Collect all tools IDs and verify they are globally unique
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_GloballyUniqueAcrossMultipleGraphs) {
  hipGraph_t graph1, graph2;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  constexpr int nodesPerGraph = 3;
  std::set<unsigned long long> allToolsIds;

  for (int i = 0; i < nodesPerGraph; i++) {
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddEmptyNode(&node, graph1, nullptr, 0));
    unsigned long long tid;
    HIP_CHECK(hipGraphNodeGetToolsId(node, &tid));
    allToolsIds.insert(tid);
  }

  for (int i = 0; i < nodesPerGraph; i++) {
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddEmptyNode(&node, graph2, nullptr, 0));
    unsigned long long tid;
    HIP_CHECK(hipGraphNodeGetToolsId(node, &tid));
    allToolsIds.insert(tid);
  }

  REQUIRE(allToolsIds.size() == 2 * nodesPerGraph);

  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify tools ID encoding holds for nodes in a cloned graph:
 *        -# Create a graph with a node, clone it
 *        -# Verify cloned node's toolsId == (clonedGraphId << 32) | localId
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_EncodingHoldsForClonedGraph) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  hipGraph_t clonedGraph;
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));

  unsigned int clonedGraphId;
  HIP_CHECK(hipGraphGetId(clonedGraph, &clonedGraphId));

  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(clonedGraph, nullptr, &numNodes));
  REQUIRE(numNodes == 1);

  hipGraphNode_t clonedNode;
  HIP_CHECK(hipGraphGetNodes(clonedGraph, &clonedNode, &numNodes));

  unsigned int clonedLocalId;
  unsigned long long clonedToolsId;
  HIP_CHECK(hipGraphNodeGetLocalId(clonedNode, &clonedLocalId));
  HIP_CHECK(hipGraphNodeGetToolsId(clonedNode, &clonedToolsId));

  unsigned long long expected =
      ((unsigned long long)clonedGraphId << 32) | clonedLocalId;
  REQUIRE(clonedToolsId == expected);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify child graph node's tools ID uses the parent graph's ID:
 *        -# Create parent and child graphs
 *        -# Add child as child graph node in parent
 *        -# Verify child graph node's toolsId encodes the parent graph's ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_ChildGraphNodeUsesParentGraphId) {
  hipGraph_t parentGraph, childGraph;
  HIP_CHECK(hipGraphCreate(&parentGraph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipGraphNode_t childEmptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&childEmptyNode, childGraph, nullptr, 0));

  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, parentGraph, nullptr, 0,
                                       childGraph));

  unsigned int parentGraphId;
  HIP_CHECK(hipGraphGetId(parentGraph, &parentGraphId));

  unsigned int localId;
  unsigned long long toolsId;
  HIP_CHECK(hipGraphNodeGetLocalId(childGraphNode, &localId));
  HIP_CHECK(hipGraphNodeGetToolsId(childGraphNode, &toolsId));

  unsigned long long expected =
      ((unsigned long long)parentGraphId << 32) | localId;
  REQUIRE(toolsId == expected);

  HIP_CHECK(hipGraphDestroy(parentGraph));
  HIP_CHECK(hipGraphDestroy(childGraph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
