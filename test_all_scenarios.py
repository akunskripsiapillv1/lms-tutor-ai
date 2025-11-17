#!/usr/bin/env python3
"""
Comprehensive Test File for All 4 Chat Service Scenarios

This script tests all 4 expected behaviors:
1. hit_raw: Cache hit without user context (should return cache_raw)
2. hit_personalized: Cache hit with user context (should personalize)
3. knowledge_base: Cache miss + RAG finds documents (should use knowledge_base)
4. llm_fallback: Cache miss + no RAG documents (should use LLM fallback)

Usage: python test_all_scenarios.py
"""
import asyncio
import sys
import os
import json
import time

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.optimized_chat_service import OptimizedChatService
from app.models.chat import ChatRequest
from app.core.logger import api_logger

class ScenarioTester:
    def __init__(self):
        self.chat_service = None
        self.results = []

    async def setup(self):
        """Initialize chat service."""
        try:
            self.chat_service = OptimizedChatService()
            await self.chat_service.connect()
            api_logger.info("🚀 Scenario Tester initialized successfully")
            return True
        except Exception as e:
            api_logger.error(f"❌ Failed to initialize: {str(e)}")
            return False

    async def test_scenario(self, name: str, description: str, request: ChatRequest, expected_source: str):
        """Test a single scenario and log results."""
        api_logger.info(f"\n🧪 Testing Scenario: {name}")
        api_logger.info(f"📝 Description: {description}")
        api_logger.info(f"❓ Query: {request.query}")
        api_logger.info(f"👤 User ID: {request.user_id}")
        api_logger.info(f"🎓 Course ID: {request.course_id}")
        api_logger.info(f"🔧 Context: {bool(request.context)} (length: {len(str(request.context)) if request.context else 0})")

        try:
            start_time = time.time()
            response = await self.chat_service.chat_completion(request)
            end_time = time.time()

            # Check if scenario matches expectation
            actual_source = response.response_source
            is_cached = response.cached
            sources_count = len(response.sources)
            latency = end_time - start_time

            # Determine if test passed
            passed = actual_source == expected_source

            # Log detailed results
            api_logger.info(f"📊 Results:")
            api_logger.info(f"   ✅ Expected: {expected_source}")
            api_logger.info(f"   🎯 Actual: {actual_source}")
            api_logger.info(f"   {'✅' if passed else '❌'} Test: {'PASSED' if passed else 'FAILED'}")
            api_logger.info(f"   💾 Cached: {is_cached}")
            api_logger.info(f"   📚 Sources: {sources_count} items")
            api_logger.info(f"   ⏱️  Latency: {latency:.2f}s")
            api_logger.info(f"   💰 Response length: {len(response.response)} chars")

            # Save result
            result = {
                "scenario": name,
                "description": description,
                "query": request.query,
                "user_id": request.user_id,
                "course_id": request.course_id,
                "has_context": bool(request.context),
                "expected_source": expected_source,
                "actual_source": actual_source,
                "passed": passed,
                "cached": is_cached,
                "sources_count": sources_count,
                "latency_ms": latency * 1000,
                "response_preview": response.response[:200] + "..." if len(response.response) > 200 else response.response
            }

            self.results.append(result)

            # Print summary to console
            print(f"\n{'='*80}")
            print(f"🧪 SCENARIO: {name}")
            print(f"{'='*80}")
            print(f"📝 Description: {description}")
            print(f"❓ Query: {request.query}")
            print(f"👤 User: {request.user_id} | 🎓 Course: {request.course_id}")
            print(f"🔧 Context: {'Yes' if request.context else 'No'}")
            print(f"📊 Results:")
            print(f"   Expected: {expected_source}")
            print(f"   Actual: {actual_source}")
            print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
            print(f"   Cached: {is_cached} | Sources: {sources_count} | Latency: {latency:.2f}s")
            print(f"   Response Preview: {result['response_preview']}")

            return passed

        except Exception as e:
            api_logger.error(f"❌ Scenario {name} failed: {str(e)}")
            error_result = {
                "scenario": name,
                "error": str(e),
                "passed": False
            }
            self.results.append(error_result)
            return False

    async def run_all_tests(self):
        """Run all 4 scenarios in sequence."""
        print(f"\n🎯 Starting Comprehensive Chat Service Testing...")
        print(f"{'='*80}")

        # Scenario 1: hit_raw - First query without context, then same query again without context
        print(f"\n📍 STEP 1: Testing cache_raw scenario")

        # 1a: First query (cache miss, should generate LLM response)
        request_1a = ChatRequest(
            query="What is ChartInstruct?",
            user_id="test-user-001",
            course_id="test-course-001",
            context=None  # NO context = should NOT personalize
        )

        await self.test_scenario(
            name="Scenario 1a: First Query (Cache Miss)",
            description="First query without context - should use LLM fallback and store in cache",
            request=request_1a,
            expected_source="knowledge_base"  # Should find Python documents via RAG
        )

        # Wait a bit for cache to store
        await asyncio.sleep(1)

        # 1b: Same query again without context (should hit cache raw)
        request_1b = ChatRequest(
            query="What is ChartInstruct?",  # EXACT same query
            user_id="test-user-001",           # SAME user
            course_id="test-course-001",         # SAME course
            context=None                         # STILL no context = cache_raw
        )

        await self.test_scenario(
            name="Scenario 1b: Cache Hit Without Context",
            description="Same query again without context - should return cache_raw (no personalization)",
            request=request_1b,
            expected_source="cache_raw"  # Should hit cache raw
        )

        await asyncio.sleep(1)

        # Scenario 2: hit_personalized - Same query with user context
        print(f"\n📍 STEP 2: Testing hit_personalized scenario")

        request_2 = ChatRequest(
            query="What is ChartInstruct?",  # SAME query
            user_id="test-user-001",           # SAME user
            course_id="test-course-001",         # SAME course
            context={                             # NOW with context = should personalize!
                "preferences": ["beginner-friendly explanations", "code examples"],
                "goals": ["learn Python basics", "build web applications"],
                "history": ["asked about variables", "asked about functions"]
            }
        )

        await self.test_scenario(
            name="Scenario 2: Cache Hit With User Context",
            description="Same query with user context - should personalize cached response",
            request=request_2,
            expected_source="cache_personalized"  # Should personalize cache hit
        )

        await asyncio.sleep(1)

        # Scenario 3: knowledge_base - New query that should match existing documents
        print(f"\n📍 STEP 3: Testing knowledge_base scenario")

        request_3 = ChatRequest(
            query="What is ChartInstruct?",  # This should match your existing documents
            user_id="test-user-001",
            course_id="test-course-001",   # Different course
            context=None                    # No context needed
        )

        await self.test_scenario(
            name="Scenario 3: RAG Knowledge Base",
            description="Query about ChartInstruct - should find existing documents in knowledge base",
            request=request_3,
            expected_source="knowledge_base"  # Should use RAG from knowledge base
        )

        await asyncio.sleep(1)

        # Scenario 4: llm_fallback - Query about completely unrelated topic
        print(f"\n📍 STEP 4: Testing llm_fallback scenario")

        request_4 = ChatRequest(
            query="How to make chocolate chip cookies?",  # Completely unrelated to ChartInstruct
            user_id="test-user-001",
            course_id="test-course-001",                  # Different course
            context=None
        )

        await self.test_scenario(
            name="Scenario 4: LLM Fallback",
            description="Query about baking - no relevant documents, should use LLM fallback",
            request=request_4,
            expected_source="llm_fallback"  # Should use LLM fallback
        )

        # Generate final report
        await self.generate_report()

    async def generate_report(self):
        """Generate comprehensive test report."""
        print(f"\n📊 FINAL TEST REPORT")
        print(f"{'='*80}")

        passed_count = sum(1 for r in self.results if r.get("passed", False))
        total_count = len([r for r in self.results if "error" not in r])

        print(f"📈 Overall Results: {passed_count}/{total_count} scenarios passed")
        print(f"🎯 Success Rate: {(passed_count/total_count*100):.1f}%")

        print(f"\n📋 Detailed Results:")
        for i, result in enumerate(self.results, 1):
            if "error" in result:
                print(f"   {i}. ❌ {result['scenario']}: ERROR - {result['error']}")
            else:
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                print(f"   {i}. {status} - {result['scenario']}")
                print(f"      Expected: {result['expected_source']} | Actual: {result['actual_source']}")
                if not result['passed']:
                    print(f"      💡 Why: Expected {result['expected_source']} but got {result['actual_source']}")

        # Special attention to cache_raw scenario
        cache_raw_result = next((r for r in self.results if "Cache Hit Without Context" in r.get("scenario", "")), None)
        if cache_raw_result:
            print(f"\n🔍 Cache Raw Analysis:")
            if cache_raw_result['passed']:
                print(f"   ✅ SUCCESS: Cache raw working correctly!")
                print(f"   📊 Query: {cache_raw_result['query']}")
                print(f"   💾 Response cached and retrieved without personalization")
                print(f"   ⚡ Latency: {cache_raw_result['latency_ms']:.1f}ms (should be fast)")
            else:
                print(f"   ❌ ISSUE: Cache raw not working as expected")
                print(f"   🎯 Got: {cache_raw_result['actual_source']} instead of cache_raw")
                print(f"   💡 Possible causes:")
                print(f"      - User context is being auto-injected somewhere")
                print(f"      - Cache logic needs adjustment")
                print(f"      - Personalization logic is too aggressive")

        # Save results to file
        report_file = f"test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {report_file}")
        print(f"\n🎯 Key Takeaways:")

        if passed_count == total_count:
            print(f"   🎉 All scenarios working perfectly!")
            print(f"   ✅ Cache system is functioning correctly")
            print(f"   ✅ RAG integration is working")
            print(f"   ✅ Personalization logic is correct")
        else:
            print(f"   ⚠️  Some scenarios need attention")
            print(f"   🔧 Review the failed scenarios above")

        # Specific guidance for cache_raw
        if cache_raw_result and not cache_raw_result['passed']:
            print(f"\n🔧 Cache Raw Debugging:")
            print(f"   1. Check if request.context is truly None in scenario 1b")
            print(f"   2. Verify _should_personalize logic in simplified_semantic_cache.py")
            print(f"   3. Check if any automatic user context is being added")

async def main():
    """Main test runner."""
    tester = ScenarioTester()

    print(f"🚀 Starting Chat Service Scenario Testing...")

    # Setup
    if not await tester.setup():
        print("❌ Failed to setup test environment")
        return

    # Run all tests
    await tester.run_all_tests()

    print(f"\n🏁 Testing completed!")

if __name__ == "__main__":
    asyncio.run(main())