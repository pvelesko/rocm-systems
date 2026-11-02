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
 * @addtogroup hipGraphNodeGetLocalId hipGraphNodeGetLocalId
 * @{
 * @ingroup GraphTest
 * `hipGraphNodeGetLocalId(hipGraphNode_t hNode, unsigned int *nodeId)` -
 * Returns the local node id of a given graph node
 */

static __global__ void emptyKernel() {}

/**
 * Test Description
 * ------------------------
 *    - Verify local node IDs start at zero and increment in creation order:
 *        -# Create a graph with multiple empty nodes
 *        -# Verify their local IDs are 0, 1, 2, 3, 4
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_UniqueWithinGraph) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  constexpr int numNodes = 5;
  hipGraphNode_t nodes[numNodes];
  unsigned int ids[numNodes];

  for (int i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphAddEmptyNode(&nodes[i], graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetLocalId(nodes[i], &ids[i]));
  }

  for (int i = 0; i < numNodes; i++) {
    REQUIRE(ids[i] == static_cast<unsigned int>(i));
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify local IDs are unique for different node types in the same graph:
 *        -# Create a graph with empty, kernel, and memset nodes
 *        -# Verify all their local IDs are distinct
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_DifferentNodeTypes) {
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

  // Memset node
  hipGraphNode_t memsetNode;
  int* devMem;
  HIP_CHECK(hipMalloc(&devMem, 1024));
  hipMemsetParams memsetParams = {};
  memsetParams.dst = devMem;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = 1;
  memsetParams.width = 1024;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                   &memsetParams));

  unsigned int emptyId, kernelId, memsetId;
  HIP_CHECK(hipGraphNodeGetLocalId(emptyNode, &emptyId));
  HIP_CHECK(hipGraphNodeGetLocalId(kernelNode, &kernelId));
  HIP_CHECK(hipGraphNodeGetLocalId(memsetNode, &memsetId));

  REQUIRE(emptyId != kernelId);
  REQUIRE(emptyId != memsetId);
  REQUIRE(kernelId != memsetId);

  HIP_CHECK(hipFree(devMem));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify local IDs are unique for nodes with dependencies:
 *        -# Create a chain of dependent empty nodes
 *        -# Verify all their local IDs are distinct
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_WithDependencies) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t nodeA, nodeB, nodeC;
  HIP_CHECK(hipGraphAddEmptyNode(&nodeA, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&nodeB, graph, &nodeA, 1));
  HIP_CHECK(hipGraphAddEmptyNode(&nodeC, graph, &nodeB, 1));

  unsigned int idA, idB, idC;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeA, &idA));
  HIP_CHECK(hipGraphNodeGetLocalId(nodeB, &idB));
  HIP_CHECK(hipGraphNodeGetLocalId(nodeC, &idC));

  REQUIRE(idA != idB);
  REQUIRE(idA != idC);
  REQUIRE(idB != idC);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphNodeGetLocalId works on a child graph node:
 *        -# Create parent and child graphs
 *        -# Add child graph as child graph node in parent
 *        -# Verify local IDs are obtainable for both
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_ChildGraphNode) {
  hipGraph_t parentGraph, childGraph;
  HIP_CHECK(hipGraphCreate(&parentGraph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipGraphNode_t childEmptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&childEmptyNode, childGraph, nullptr, 0));

  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, parentGraph, nullptr, 0,
                                       childGraph));

  unsigned int childGraphNodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(childGraphNode, &childGraphNodeId));

  unsigned int childEmptyNodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(childEmptyNode, &childEmptyNodeId));

  REQUIRE(childGraphNodeId == 0);
  REQUIRE(childEmptyNodeId == 0);

  HIP_CHECK(hipGraphDestroy(parentGraph));
  HIP_CHECK(hipGraphDestroy(childGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify node IDs are unique in a graph obtained from stream capture:
 *        -# Capture a stream with multiple kernel launches
 *        -# Extract nodes and verify all local IDs are unique
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_CapturedStream) {
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

  std::set<unsigned int> idSet;
  for (size_t i = 0; i < numNodes; i++) {
    unsigned int nodeId;
    HIP_CHECK(hipGraphNodeGetLocalId(nodes[i], &nodeId));
    idSet.insert(nodeId);
  }

  REQUIRE(idSet.size() == numNodes);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphNodeGetLocalId works on a host node:
 *        -# Create a graph with a host node
 *        -# Verify hipGraphNodeGetLocalId returns successfully
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
static void dummyHostFn(void*) {}

HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_HostNode) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipHostNodeParams hostParams = {};
  hostParams.fn = dummyHostFn;
  hostParams.userData = nullptr;

  hipGraphNode_t hostNode;
  HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

  unsigned int nodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(hostNode, &nodeId));
  REQUIRE(nodeId == 0);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphNodeGetLocalId works on an event record node:
 *        -# Create a graph with an event record node
 *        -# Verify hipGraphNodeGetLocalId returns successfully
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_EventRecordNode) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipGraphNode_t eventRecordNode;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventRecordNode, graph, nullptr, 0,
                                        event));

  unsigned int nodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(eventRecordNode, &nodeId));
  REQUIRE(nodeId == 0);

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphNodeGetLocalId works on an event wait node:
 *        -# Create a graph with an event wait node
 *        -# Verify hipGraphNodeGetLocalId returns successfully
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_EventWaitNode) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipGraphNode_t eventWaitNode;
  HIP_CHECK(hipGraphAddEventWaitNode(&eventWaitNode, graph, nullptr, 0,
                                      event));

  unsigned int nodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(eventWaitNode, &nodeId));
  REQUIRE(nodeId == 0);

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphNodeGetLocalId works on a memcpy node:
 *        -# Create a graph with a 1D memcpy node
 *        -# Verify hipGraphNodeGetLocalId returns successfully
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_MemcpyNode) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  float *dSrc, *dDst;
  HIP_CHECK(hipMalloc(&dSrc, 1024));
  HIP_CHECK(hipMalloc(&dDst, 1024));

  hipGraphNode_t memcpyNode;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, dDst,
                                     dSrc, 1024, hipMemcpyDeviceToDevice));

  unsigned int nodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(memcpyNode, &nodeId));
  REQUIRE(nodeId == 0);

  HIP_CHECK(hipFree(dSrc));
  HIP_CHECK(hipFree(dDst));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify cloned graph preserves local node IDs from the original:
 *        -# Create a graph with three nodes, record their local IDs
 *        -# Clone the graph
 *        -# Verify cloned nodes have the same local IDs as originals
 *        -# Add a node to the clone and verify its ID follows the cloned sequence
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_ClonedGraphPreservesIds) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  constexpr int numNodes = 3;
  hipGraphNode_t origNodes[numNodes];
  unsigned int origIds[numNodes];
  for (int i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphAddEmptyNode(&origNodes[i], graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetLocalId(origNodes[i], &origIds[i]));
  }

  hipGraph_t clonedGraph;
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));

  size_t clonedCount = 0;
  HIP_CHECK(hipGraphGetNodes(clonedGraph, nullptr, &clonedCount));
  REQUIRE(clonedCount == numNodes);

  std::vector<hipGraphNode_t> clonedNodes(clonedCount);
  HIP_CHECK(hipGraphGetNodes(clonedGraph, clonedNodes.data(), &clonedCount));

  std::set<unsigned int> origIdSet(origIds, origIds + numNodes);
  std::set<unsigned int> clonedIdSet;
  for (size_t i = 0; i < clonedCount; i++) {
    unsigned int nodeId;
    HIP_CHECK(hipGraphNodeGetLocalId(clonedNodes[i], &nodeId));
    clonedIdSet.insert(nodeId);
  }

  REQUIRE(clonedIdSet == origIdSet);

  hipGraphNode_t appendedNode;
  HIP_CHECK(hipGraphAddEmptyNode(&appendedNode, clonedGraph, nullptr, 0));
  unsigned int appendedId;
  HIP_CHECK(hipGraphNodeGetLocalId(appendedNode, &appendedId));
  REQUIRE(appendedId == static_cast<unsigned int>(numNodes));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify local node IDs restart for each graph:
 *        -# Create two graphs with two nodes each
 *        -# Verify both graphs assign local IDs 0 and 1
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_RestartsForEachGraph) {
  hipGraph_t graph1, graph2;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  hipGraphNode_t node1a, node1b, node2a, node2b;
  HIP_CHECK(hipGraphAddEmptyNode(&node1a, graph1, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&node1b, graph1, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&node2a, graph2, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&node2b, graph2, nullptr, 0));

  unsigned int id1a, id1b, id2a, id2b;
  HIP_CHECK(hipGraphNodeGetLocalId(node1a, &id1a));
  HIP_CHECK(hipGraphNodeGetLocalId(node1b, &id1b));
  HIP_CHECK(hipGraphNodeGetLocalId(node2a, &id2a));
  HIP_CHECK(hipGraphNodeGetLocalId(node2b, &id2b));

  REQUIRE(id1a == 0);
  REQUIRE(id1b == 1);
  REQUIRE(id2a == 0);
  REQUIRE(id2b == 1);

  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify node local IDs are stable after removing a sibling node:
 *        -# Create a graph with three nodes (A, B, C)
 *        -# Remove node A
 *        -# Verify node B and C local IDs are unchanged
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_StableAfterSiblingRemoval) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t nodeA, nodeB, nodeC;
  HIP_CHECK(hipGraphAddEmptyNode(&nodeA, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&nodeB, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&nodeC, graph, nullptr, 0));

  unsigned int idB_before, idC_before;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeB, &idB_before));
  HIP_CHECK(hipGraphNodeGetLocalId(nodeC, &idC_before));

  HIP_CHECK(hipGraphDestroyNode(nodeA));

  unsigned int idB_after, idC_after;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeB, &idB_after));
  HIP_CHECK(hipGraphNodeGetLocalId(nodeC, &idC_after));

  REQUIRE(idB_before == idB_after);
  REQUIRE(idC_before == idC_after);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify that local node IDs are not reused after node removal:
 *        -# Create a graph with two nodes (A, B)
 *        -# Remove node A
 *        -# Add a new node C
 *        -# Verify C gets a new local ID (does not reuse A's ID)
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_NoIdReuseAfterRemoval) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t nodeA, nodeB;
  HIP_CHECK(hipGraphAddEmptyNode(&nodeA, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&nodeB, graph, nullptr, 0));

  unsigned int idA, idB;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeA, &idA));
  HIP_CHECK(hipGraphNodeGetLocalId(nodeB, &idB));

  HIP_CHECK(hipGraphDestroyNode(nodeA));

  hipGraphNode_t nodeC;
  HIP_CHECK(hipGraphAddEmptyNode(&nodeC, graph, nullptr, 0));

  unsigned int idC;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeC, &idC));

  // New node should not reuse the removed node's ID
  REQUIRE(idC != idA);
  REQUIRE(idC != idB);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify nodes inside an embedded child graph have valid, unique local IDs:
 *        -# Create a child graph with two nodes
 *        -# Add child graph as child node in parent
 *        -# Retrieve embedded child graph copy
 *        -# Verify its nodes have unique local IDs
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_EmbeddedChildGraphNodes) {
  hipGraph_t parentGraph, childGraph;
  HIP_CHECK(hipGraphCreate(&parentGraph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipGraphNode_t childNode1, childNode2;
  HIP_CHECK(hipGraphAddEmptyNode(&childNode1, childGraph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&childNode2, childGraph, &childNode1, 1));

  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, parentGraph, nullptr, 0,
                                       childGraph));

  hipGraph_t embeddedChild;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &embeddedChild));

  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(embeddedChild, nullptr, &numNodes));
  REQUIRE(numNodes == 2);

  std::vector<hipGraphNode_t> embeddedNodes(numNodes);
  HIP_CHECK(hipGraphGetNodes(embeddedChild, embeddedNodes.data(), &numNodes));

  std::set<unsigned int> idSet;
  for (size_t i = 0; i < numNodes; i++) {
    unsigned int nodeId;
    HIP_CHECK(hipGraphNodeGetLocalId(embeddedNodes[i], &nodeId));
    idSet.insert(nodeId);
  }

  REQUIRE(idSet.size() == numNodes);

  HIP_CHECK(hipGraphDestroy(parentGraph));
  HIP_CHECK(hipGraphDestroy(childGraph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
