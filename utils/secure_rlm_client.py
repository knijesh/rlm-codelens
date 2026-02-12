"""
Secure RLM Client with Pre-flight Cost Estimation
Production-ready RLM wrapper with security, performance, and cost controls
"""

import re
import json
import time
import hashlib
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
import threading
from functools import wraps
import tiktoken


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CostEstimate:
    """Pre-flight cost estimation"""

    input_tokens: int
    output_tokens: int  # Estimated
    estimated_cost: float
    model: str
    confidence: float  # 0-1, how confident in estimate

    def __str__(self):
        return f"{self.input_tokens} input + {self.output_tokens} output tokens = ${self.estimated_cost:.4f}"


@dataclass
class RLMResult:
    """Structured RLM result"""

    success: bool
    response: str
    cost: float
    tokens_used: int
    duration_ms: float
    cache_hit: bool
    error: Optional[str] = None


class PromptSanitizer:
    """Sanitizes user inputs to prevent prompt injection"""

    # Patterns that could indicate prompt injection
    DANGEROUS_PATTERNS = [
        r"ignore previous instructions",
        r"ignore all prior",
        r"disregard.*instructions",
        r"you are now.*assistant",
        r"system.*prompt",
        r"<script",
        r"javascript:",
        r"\\x[0-9a-fA-F]{2}",  # Hex encoding
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitize text to prevent injection attacks"""
        if not isinstance(text, str):
            text = str(text)

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

        # Check for dangerous patterns
        text_lower = text.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower):
                # Replace with safe placeholder
                text = text[:100] + "...[CONTENT SANITIZED]"
                break

        # Escape special characters
        text = text.replace('"', '\\"').replace("'", "\\'")

        # Limit length to prevent token explosion
        max_length = 5000  # ~1250 tokens
        if len(text) > max_length:
            text = text[:max_length] + "...[TRUNCATED]"

        return text

    @classmethod
    def validate_json_safety(cls, data: Any) -> bool:
        """Check if data is safe to serialize to JSON"""
        try:
            json_str = json.dumps(data)
            # Check for excessive nesting
            if json_str.count("{") > 50 or json_str.count("[") > 50:
                return False
            return True
        except (TypeError, ValueError):
            return False


class TokenCounter:
    """Accurate token counting for cost estimation"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages"""
        total = 0
        for message in messages:
            # Base tokens per message
            total += 4

            # Content tokens
            if "content" in message:
                total += self.count_tokens(message["content"])

            # Role tokens
            if "role" in message:
                total += self.count_tokens(message["role"])

        # Base tokens for response
        total += 2

        return total


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    print("Circuit breaker: Testing recovery...")
                else:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker is OPEN - service unavailable"
                    )

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        "Circuit breaker: Too many test calls"
                    )
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                print("Circuit breaker: CLOSED - service recovered")

    def _on_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker: OPEN - recovery failed")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker: OPEN after {self.failure_count} failures")


class CircuitBreakerOpenError(Exception):
    """Exception when circuit breaker is open"""

    pass


class CostEstimator:
    """Pre-flight cost estimation and budget enforcement"""

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-16k": {"input": 1.0, "output": 2.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.token_counter = TokenCounter(model)

    def estimate_chat_cost(
        self, prompt: str, expected_output_tokens: int = 500, confidence: float = 0.8
    ) -> CostEstimate:
        """Estimate cost for a chat completion"""
        input_tokens = self.token_counter.count_tokens(prompt)

        pricing = self.PRICING.get(self.model, {"input": 0.50, "output": 1.50})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (expected_output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=expected_output_tokens,
            estimated_cost=total_cost,
            model=self.model,
            confidence=confidence,
        )

    def estimate_embeddings_cost(self, texts: List[str]) -> CostEstimate:
        """Estimate cost for embeddings"""
        total_tokens = sum(self.token_counter.count_tokens(text) for text in texts)

        pricing = self.PRICING.get(self.model, {"input": 0.02, "output": 0.0})
        cost = (total_tokens / 1_000_000) * pricing["input"]

        return CostEstimate(
            input_tokens=total_tokens,
            output_tokens=0,
            estimated_cost=cost,
            model=self.model,
            confidence=0.95,
        )

    @classmethod
    def compare_approaches(
        cls,
        rlm_calls: int,
        rlm_avg_tokens: int,
        non_rlm_calls: int,
        non_rlm_avg_tokens: int,
        model: str = "gpt-3.5-turbo",
    ) -> Dict[str, Any]:
        """Compare RLM vs Non-RLM cost"""
        pricing = cls.PRICING.get(model, {"input": 0.50, "output": 1.50})

        # RLM cost (typically uses cheaper model for sub-calls)
        rlm_input = rlm_calls * rlm_avg_tokens * 0.7  # 70% input
        rlm_output = rlm_calls * rlm_avg_tokens * 0.3  # 30% output
        rlm_cost = (
            rlm_input / 1_000_000 * pricing["input"]
            + rlm_output / 1_000_000 * pricing["output"]
        )

        # Non-RLM cost (processes everything at once - higher quality model)
        non_rlm_input = non_rlm_calls * non_rlm_avg_tokens * 0.7
        non_rlm_output = non_rlm_calls * non_rlm_avg_tokens * 0.3
        non_rlm_cost = (
            non_rlm_input / 1_000_000 * pricing["input"]
            + non_rlm_output / 1_000_000 * pricing["output"]
        )

        savings = non_rlm_cost - rlm_cost
        savings_pct = (savings / non_rlm_cost * 100) if non_rlm_cost > 0 else 0

        return {
            "rlm_cost": rlm_cost,
            "non_rlm_cost": non_rlm_cost,
            "savings": savings,
            "savings_percentage": savings_pct,
            "rlm_calls": rlm_calls,
            "non_rlm_calls": non_rlm_calls,
            "cost_efficient": rlm_cost < non_rlm_cost,
        }


class SecureRLMClient:
    """
    Production-ready secure RLM client with:
    - Pre-flight cost estimation
    - Budget enforcement BEFORE API calls
    - Circuit breaker pattern
    - Input sanitization
    - Result caching
    - Retry logic
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        budget_limit: float = 50.0,
        enable_caching: bool = True,
        enable_circuit_breaker: bool = True,
        max_retries: int = 3,
    ):
        self.model = model
        self.budget_limit = budget_limit
        self.total_spent = 0.0
        self.enable_caching = enable_caching
        self.max_retries = max_retries

        # Initialize components
        self.cost_estimator = CostEstimator(model)
        self.token_counter = TokenCounter(model)
        self.cache: Dict[str, Any] = {} if enable_caching else None
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        # Import RLM only when needed (lazy loading)
        self._rlm = None

    @property
    def rlm(self):
        """Lazy load RLM"""
        if self._rlm is None:
            try:
                from rlm import RLM

                self._rlm = RLM(
                    backend="openai", backend_kwargs={"model_name": self.model}
                )
            except ImportError:
                print("⚠️  RLM library not installed. Using fallback mode.")
                self._rlm = None
        return self._rlm

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _check_budget(self, estimated_cost: float) -> bool:
        """Check if call is within budget BEFORE making it"""
        projected_total = self.total_spent + estimated_cost

        if projected_total > self.budget_limit:
            print(f"\n⛔ BUDGET EXCEEDED!")
            print(f"   Current: ${self.total_spent:.2f}")
            print(f"   This call: ${estimated_cost:.4f}")
            print(f"   Would total: ${projected_total:.2f}")
            print(f"   Budget: ${self.budget_limit:.2f}")
            return False

        if projected_total > self.budget_limit * 0.8:
            print(
                f"\n⚠️  BUDGET WARNING: {projected_total / self.budget_limit * 100:.0f}% used"
            )

        return True

    def completion(
        self,
        prompt: str,
        expected_output_tokens: int = 500,
        use_cache: bool = True,
        skip_if_over_budget: bool = True,
    ) -> RLMResult:
        """
        Execute RLM completion with full cost control

        Args:
            prompt: The prompt to send
            expected_output_tokens: Expected response size
            use_cache: Whether to use caching
            skip_if_over_budget: Skip call if over budget

        Returns:
            RLMResult with success status and metadata
        """
        start_time = time.time()

        # Step 1: Sanitize input
        safe_prompt = PromptSanitizer.sanitize(prompt)

        # Step 2: Check cache
        if use_cache and self.cache is not None:
            cache_key = self._get_cache_key(safe_prompt)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                return RLMResult(
                    success=True,
                    response=cached["response"],
                    cost=0.0,
                    tokens_used=cached["tokens"],
                    duration_ms=0.0,
                    cache_hit=True,
                )

        # Step 3: Pre-flight cost estimation
        cost_estimate = self.cost_estimator.estimate_chat_cost(
            safe_prompt, expected_output_tokens
        )

        print(f"\n💰 COST ESTIMATE: {cost_estimate}")

        # Step 4: Budget check BEFORE API call
        if not self._check_budget(cost_estimate.estimated_cost):
            if skip_if_over_budget:
                return RLMResult(
                    success=False,
                    response="",
                    cost=0.0,
                    tokens_used=0,
                    duration_ms=0.0,
                    cache_hit=False,
                    error="Budget limit exceeded",
                )
            else:
                raise BudgetExceededError(
                    f"Call would exceed budget: ${cost_estimate.estimated_cost:.4f}"
                )

        # Step 5: Execute with circuit breaker and retries
        try:
            result = self._execute_with_retries(safe_prompt, cost_estimate, start_time)

            # Update budget tracking
            self.total_spent += result.cost

            # Cache result
            if use_cache and self.cache is not None and result.success:
                cache_key = self._get_cache_key(safe_prompt)
                self.cache[cache_key] = {
                    "response": result.response,
                    "tokens": result.tokens_used,
                    "timestamp": time.time(),
                }

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return RLMResult(
                success=False,
                response="",
                cost=0.0,
                tokens_used=0,
                duration_ms=duration_ms,
                cache_hit=False,
                error=str(e),
            )

    def _execute_with_retries(
        self, prompt: str, cost_estimate: CostEstimate, start_time: float
    ) -> RLMResult:
        """Execute API call with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Use circuit breaker if enabled
                if self.circuit_breaker:
                    result = self.circuit_breaker.call(self._make_api_call, prompt)
                else:
                    result = self._make_api_call(prompt)

                duration_ms = (time.time() - start_time) * 1000

                # Calculate actual cost
                tokens_used = cost_estimate.input_tokens + cost_estimate.output_tokens
                actual_cost = cost_estimate.estimated_cost

                return RLMResult(
                    success=True,
                    response=result,
                    cost=actual_cost,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    cache_hit=False,
                )

            except Exception as e:
                last_error = e
                wait_time = (2**attempt) + (attempt * 0.1)  # Exponential backoff
                print(
                    f"   ⚠️  Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        # All retries failed
        duration_ms = (time.time() - start_time) * 1000
        raise Exception(f"All {self.max_retries} retries failed: {last_error}")

    def _make_api_call(self, prompt: str) -> str:
        """Make actual API call"""
        if self.rlm is None:
            # Fallback mode - simulate response
            return self._fallback_response(prompt)

        # Use RLM library
        result = self.rlm.completion(prompt)
        return result.response if hasattr(result, "response") else str(result)

    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when RLM not available"""
        return json.dumps(
            {
                "topic": "Fallback Analysis",
                "description": "RLM library not available - using fallback",
                "category": "Other",
                "key_terms": ["fallback"],
                "grouping_reason": "No RLM available",
            }
        )

    def get_budget_summary(self) -> Dict[str, Any]:
        """Get current budget status"""
        return {
            "budget_limit": self.budget_limit,
            "total_spent": self.total_spent,
            "remaining": self.budget_limit - self.total_spent,
            "percentage_used": (self.total_spent / self.budget_limit * 100),
            "cache_size": len(self.cache) if self.cache else 0,
            "circuit_state": self.circuit_breaker.state.value
            if self.circuit_breaker
            else "disabled",
        }

    def print_budget_summary(self):
        """Print budget summary"""
        summary = self.get_budget_summary()
        print("\n" + "=" * 60)
        print("BUDGET SUMMARY")
        print("=" * 60)
        print(f"Limit: ${summary['budget_limit']:.2f}")
        print(f"Spent: ${summary['total_spent']:.2f}")
        print(f"Remaining: ${summary['remaining']:.2f}")
        print(f"Used: {summary['percentage_used']:.1f}%")
        print(f"Cache entries: {summary['cache_size']}")
        print("=" * 60 + "\n")


class BudgetExceededError(Exception):
    """Exception when budget would be exceeded"""

    pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SECURE RLM CLIENT - DEMONSTRATION")
    print("=" * 70)

    # Initialize client with tight budget for demo
    client = SecureRLMClient(
        model="gpt-3.5-turbo",
        budget_limit=1.0,  # $1.00 budget for demo
        enable_caching=True,
        enable_circuit_breaker=True,
    )

    # Example prompts
    test_prompts = [
        "Analyze this cluster of PyTorch issues about CUDA memory management",
        "Same prompt again (should be cached)",
        "A" * 10000,  # Large prompt to test sanitization
    ]

    print("\n🧪 Testing Secure RLM Client...")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {prompt[:50]}...")
        print("=" * 70)

        result = client.completion(prompt, expected_output_tokens=300, use_cache=True)

        print(f"Success: {result.success}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"Tokens: {result.tokens_used}")
        print(f"Duration: {result.duration_ms:.0f}ms")
        print(f"Cache hit: {result.cache_hit}")
        if result.error:
            print(f"Error: {result.error}")

    # Print final budget summary
    client.print_budget_summary()

    # Compare RLM vs Non-RLM
    print("\n📊 RLM vs Non-RLM Cost Comparison")
    print("=" * 70)

    comparison = CostEstimator.compare_approaches(
        rlm_calls=100,
        rlm_avg_tokens=800,
        non_rlm_calls=10,
        non_rlm_avg_tokens=8000,
        model="gpt-3.5-turbo",
    )

    print(f"RLM approach: ${comparison['rlm_cost']:.2f}")
    print(f"Non-RLM approach: ${comparison['non_rlm_cost']:.2f}")
    print(
        f"Savings: ${comparison['savings']:.2f} ({comparison['savings_percentage']:.1f}%)"
    )
    print(f"Cost efficient: {'✅ Yes' if comparison['cost_efficient'] else '❌ No'}")
