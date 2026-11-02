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
 * @addtogroup hipGraphGetId hipGraphGetId
 * @{
 * @ingroup GraphTest
 * `hipGraphGetId(hipGraph_t hGraph, unsigned int *graphID)` -
 * Returns the id of a given graph
 */

static __global__ void emptyKernel() {}

/**
 * Test Description
 * ------------------------
 *    - Verify graph IDs use a positive, consecutive creation sequence:
 *        -# Create multiple graphs
 *        -# Verify the first ID is positive and each subsequent ID increments by one
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphGetId_Positive_UniqueAcrossGraphs) {
  constexpr int numGraphs = 5;
  hipGraph_t graphs[numGraphs];
  unsigned int ids[numGraphs];

  for (int i = 0; i < numGraphs; i++) {
    HIP_CHECK(hipGraphCreate(&graphs[i], 0));
    HIP_CHECK(hipGraphGetId(graphs[i], &ids[i]));
  }

  REQUIRE(ids[0] >= 1);
  for (int i = 1; i < numGraphs; i++) {
    REQUIRE(ids[i] == ids[i - 1] + 1);
  }

  for (int i = 0; i < numGraphs; i++) {
    HIP_CHECK(hipGraphDestroy(graphs[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Verify that a cloned graph receives a different ID:
 *        -# Create a graph with a node
 *        -# Clone the graph
 *        -# Verify original and clone have different IDs
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphGetId_Positive_ClonedGraphHasDifferentId) {
  hipGraph_t graph, clonedGraph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t emptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));

  HIP_CHECK(hipGraphClone(&clonedGraph, graph));

  unsigned int origId = 0, clonedId = 0;
  HIP_CHECK(hipGraphGetId(graph, &origId));
  HIP_CHECK(hipGraphGetId(clonedGraph, &clonedId));

  REQUIRE(origId != clonedId);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify hipGraphGetId works with a graph obtained from stream capture:
 *        -# Begin stream capture, launch a kernel, end capture
 *        -# Verify hipGraphGetId succeeds on the captured graph
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphGetId_Positive_CapturedStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  emptyKernel<<<1, 1, 0, stream>>>();

  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  unsigned int graphId = 0;
  HIP_CHECK(hipGraphGetId(graph, &graphId));
  REQUIRE(graphId >= 1);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Verify graph ID is stable after removing all nodes:
 *        -# Create a graph, get ID
 *        -# Add a node, get ID (should be same)
 *        -# Remove the node, get ID (should still be same)
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphGetId_Positive_StableAfterNodeRemoval) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  unsigned int idEmpty;
  HIP_CHECK(hipGraphGetId(graph, &idEmpty));

  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  unsigned int idWithNode;
  HIP_CHECK(hipGraphGetId(graph, &idWithNode));

  HIP_CHECK(hipGraphDestroyNode(node));

  unsigned int idAfterRemoval;
  HIP_CHECK(hipGraphGetId(graph, &idAfterRemoval));

  REQUIRE(idEmpty == idWithNode);
  REQUIRE(idWithNode == idAfterRemoval);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify embedded child graph has a different ID from the original:
 *        -# Create parent and child graphs
 *        -# Add child as child graph node
 *        -# Retrieve the embedded child graph copy via hipGraphChildGraphNodeGetGraph
 *        -# Verify all three graph IDs (parent, original child, embedded child) are different
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphGetId_Positive_EmbeddedChildGraphHasDifferentId) {
  hipGraph_t parentGraph, childGraph;
  HIP_CHECK(hipGraphCreate(&parentGraph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipGraphNode_t childNode;
  HIP_CHECK(hipGraphAddEmptyNode(&childNode, childGraph, nullptr, 0));

  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, parentGraph, nullptr, 0,
                                       childGraph));

  hipGraph_t embeddedChildGraph;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(childGraphNode, &embeddedChildGraph));

  unsigned int parentId, originalChildId, embeddedChildId;
  HIP_CHECK(hipGraphGetId(parentGraph, &parentId));
  HIP_CHECK(hipGraphGetId(childGraph, &originalChildId));
  HIP_CHECK(hipGraphGetId(embeddedChildGraph, &embeddedChildId));

  REQUIRE(parentId != originalChildId);
  REQUIRE(parentId != embeddedChildId);
  REQUIRE(originalChildId != embeddedChildId);

  HIP_CHECK(hipGraphDestroy(parentGraph));
  HIP_CHECK(hipGraphDestroy(childGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify graph IDs are not reused after destroy and recreate:
 *        -# Create a graph, get its ID, destroy it
 *        -# Create a new graph
 *        -# Verify the new graph has a different ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphGetId_Positive_NoReuseAfterDestroy) {
  hipGraph_t graph1;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  unsigned int id1;
  HIP_CHECK(hipGraphGetId(graph1, &id1));
  HIP_CHECK(hipGraphDestroy(graph1));

  hipGraph_t graph2;
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  unsigned int id2;
  HIP_CHECK(hipGraphGetId(graph2, &id2));
  HIP_CHECK(hipGraphDestroy(graph2));

  REQUIRE(id1 != id2);
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
