import random
import math
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# 1. ENUMS AND DATA CLASSES
# ============================================================================

class TestType(Enum):
    """Types of primality tests available."""
    SOLOVAY_STRASSEN = "Solovay-Strassen"
    MILLER_RABIN = "Miller-Rabin"
    FERMAT = "Fermat"
    BAILLIE_PSW = "Baillie-PSW"
    LUCAS_LEHMER = "Lucas-Lehmer (Mersenne only)"
    TRIAL_DIVISION = "Trial Division"
    AKS = "AKS (Deterministic)"
    
class Result(Enum):
    """Possible test results."""
    PRIME = "Prime"
    COMPOSITE = "Composite"
    PROBABLY_PRIME = "Probably Prime"
    ERROR = "Error"

@dataclass
class TestResult:
    """Result of a single test."""
    test_type: TestType
    result: Result
    certainty: float  # Probability of being correct (0-1)
    execution_time: float  # Seconds
    iterations: int = 1
    message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'test_type': self.test_type.value,
            'result': self.result.value,
            'certainty': self.certainty,
            'execution_time': self.execution_time,
            'iterations': self.iterations,
            'message': self.message
        }

@dataclass
class NumberAnalysis:
    """Complete analysis of a number across all tests."""
    number: int
    is_prime: Optional[bool] = None
    test_results: List[TestResult] = None
    overall_verdict: str = ""
    total_time: float = 0.0
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'number': self.number,
            'is_prime': self.is_prime,
            'overall_verdict': self.overall_verdict,
            'total_time': self.total_time,
            'test_results': [r.to_dict() for r in self.test_results]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

# ============================================================================
# 2. CORE NUMBER THEORY FUNCTIONS
# ============================================================================

class NumberTheory:
    """Static class containing number theory utilities."""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Euclidean algorithm for greatest common divisor."""
        while b != 0:
            a, b = b, a % b
        return abs(a)
    
    @staticmethod
    def mod_pow(base: int, exp: int, mod: int) -> int:
        """Fast modular exponentiation using binary method."""
        if mod == 1:
            return 0
        result = 1
        base = base % mod
        
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp >>= 1
        
        return result
    
    @staticmethod
    def is_perfect_square(n: int) -> bool:
        """Check if n is a perfect square."""
        if n < 0:
            return False
        if n == 0:
            return True
        
        # Integer square root using Newton's method
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        
        return x * x == n
    
    @staticmethod
    def jacobi_symbol(a: int, n: int) -> int:
        """
        Compute Jacobi symbol (a/n).
        Returns 0, 1, or -1.
        """
        if n <= 0 or n % 2 == 0:
            raise ValueError("n must be odd positive integer")
        
        a = a % n
        if a == 0:
            return 0
        
        result = 1
        while a != 0:
            while a % 2 == 0:
                a //= 2
                n_mod_8 = n % 8
                if n_mod_8 in (3, 5):
                    result = -result
            a, n = n, a
            if a % 4 == 3 and n % 4 == 3:
                result = -result
            a %= n
        
        return result if n == 1 else 0

    @staticmethod
    def euler_phi(n: int) -> int:
        """Compute Euler's totient function phi(n)."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    @staticmethod
    def is_perfect_power(n: int) -> bool:
        """Checks if n is a perfect power a^b for b > 1."""
        if n < 2:
            return False
        for b in range(2, n.bit_length()):
            # Binary search for a such that a^b = n
            low = 2
            high = 2**(n.bit_length() // b + 1)
            while low <= high:
                mid = (low + high) // 2
                p = pow(mid, b)
                if p == n:
                    return True
                if p < n:
                    low = mid + 1
                else:
                    high = mid - 1
        return False

# ============================================================================
# 3. PRIMALITY TEST IMPLEMENTATIONS
# ============================================================================

class PrimalityTests:
    """Main class containing all primality test implementations."""
    
    def __init__(self):
        self.nt = NumberTheory()
        
    # ------------------------------------------------------------------------
    # 3.1 Solovay-Strassen Test
    # ------------------------------------------------------------------------
    
    def solovay_strassen(self, n: int, iterations: int = 10) -> Tuple[Result, float, str]:
        """
        Solovay-Strassen primality test.
        
        Algebraic basis: Euler's criterion for quadratic residues.
        Error probability ≤ 2^(-iterations)
        """
        start_time = time.perf_counter()
        
        # Base cases
        if n < 2:
            return Result.COMPOSITE, 0.0, "Numbers less than 2 are composite"
        if n == 2:
            return Result.PRIME, 1.0, "2 is prime"
        if n % 2 == 0:
            return Result.COMPOSITE, 0.0, "Even numbers > 2 are composite"
        if n == 3:
            return Result.PRIME, 1.0, "3 is prime"
        
        for i in range(iterations):
            a = random.randint(2, n - 2)
            
            # Check gcd
            if self.nt.gcd(a, n) > 1:
                execution_time = time.perf_counter() - start_time
                return Result.COMPOSITE, 1.0, f"Found factor gcd({a}, {n}) > 1"
            
            # Compute Jacobi symbol
            jacobi = self.nt.jacobi_symbol(a, n)
            if jacobi < 0:
                jacobi += n  # Make positive for comparison
            
            # Compute a^((n-1)/2) mod n
            exp_result = self.nt.mod_pow(a, (n - 1) // 2, n)
            
            # Euler's criterion check
            if jacobi == 0 or jacobi != exp_result:
                execution_time = time.perf_counter() - start_time
                certainty = 1 - (0.5 ** (i + 1))
                return Result.COMPOSITE, certainty, f"Failed Euler's criterion with base {a}"
        
        execution_time = time.perf_counter() - start_time
        certainty = 1 - (0.5 ** iterations)
        return Result.PROBABLY_PRIME, certainty, f"Passed {iterations} iterations"
    
    # ------------------------------------------------------------------------
    # 3.2 Miller-Rabin Test
    # ------------------------------------------------------------------------
    
    def miller_rabin(self, n: int, iterations: int = 10) -> Tuple[Result, float, str]:
        """
        Miller-Rabin primality test.
        
        Algebraic basis: Properties of square roots of 1 in finite fields.
        Error probability ≤ 4^(-iterations)
        """
        start_time = time.perf_counter()
        
        # Base cases
        if n < 2:
            return Result.COMPOSITE, 0.0, "Numbers less than 2 are composite"
        if n == 2 or n == 3:
            return Result.PRIME, 1.0, f"{n} is prime"
        if n % 2 == 0:
            return Result.COMPOSITE, 0.0, "Even numbers > 2 are composite"
        
        # Write n-1 = d * 2^s
        s = 0
        d = n - 1
        while d % 2 == 0:
            d //= 2
            s += 1
        
        for i in range(iterations):
            a = random.randint(2, n - 2)
            x = self.nt.mod_pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            composite = True
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    composite = False
                    break
            
            if composite:
                execution_time = time.perf_counter() - start_time
                certainty = 1 - (0.25 ** (i + 1))
                return Result.COMPOSITE, certainty, f"Found witness {a}"
        
        execution_time = time.perf_counter() - start_time
        certainty = 1 - (0.25 ** iterations)
        return Result.PROBABLY_PRIME, certainty, f"Passed {iterations} iterations"
    
    # ------------------------------------------------------------------------
    # 3.3 Fermat Test
    # ------------------------------------------------------------------------
    
    def fermat(self, n: int, iterations: int = 10) -> Tuple[Result, float, str]:
        """
        Fermat primality test.
        
        Algebraic basis: Fermat's Little Theorem.
        Warning: Fails for Carmichael numbers.
        """
        start_time = time.perf_counter()
        
        # Base cases
        if n < 2:
            return Result.COMPOSITE, 0.0, "Numbers less than 2 are composite"
        if n == 2:
            return Result.PRIME, 1.0, "2 is prime"
        if n % 2 == 0:
            return Result.COMPOSITE, 0.0, "Even numbers > 2 are composite"
        if n == 3:
            return Result.PRIME, 1.0, "3 is prime"
        
        # Known Carmichael numbers
        carmichael = {561, 1105, 1729, 2465, 2821, 6601, 8911, 10585}
        
        for i in range(iterations):
            a = random.randint(2, n - 2)
            
            if self.nt.gcd(a, n) != 1:
                execution_time = time.perf_counter() - start_time
                return Result.COMPOSITE, 1.0, f"Found factor gcd({a}, {n}) > 1"
            
            if self.nt.mod_pow(a, n - 1, n) != 1:
                execution_time = time.perf_counter() - start_time
                certainty = 1 - (0.5 ** (i + 1))
                return Result.COMPOSITE, certainty, f"Failed Fermat's test with base {a}"
        
        execution_time = time.perf_counter() - start_time
        
        # Special warning for Carmichael numbers
        if n in carmichael:
            certainty = 0.0  # Actually composite
            return Result.PROBABLY_PRIME, certainty, f"WARNING: {n} is a Carmichael number! Fermat test is unreliable."
        
        certainty = 1 - (0.5 ** iterations)
        return Result.PROBABLY_PRIME, certainty, f"Passed {iterations} iterations (unreliable for Carmichael numbers)"
    
    # ------------------------------------------------------------------------
    # 3.4 Lucas Sequence Helper for Baillie-PSW
    # ------------------------------------------------------------------------
    
    def _lucas_sequence_mod(self, n: int, D: int) -> bool:
        """
        Compute Lucas sequence U_{n+1} mod n.
        Returns True if U_{n+1} ≡ 0 mod n (Lucas probable prime).
        """
        P = 1
        Q = (1 - D) // 4
        
        # Helper for division by 2 modulo odd n
        def div2_mod(x: int, mod: int) -> int:
            return (x // 2) if x % 2 == 0 else ((x + mod) // 2) % mod
        
        # Fast doubling algorithm for Lucas sequences
        k = n + 1
        U, V = 1, P % n  # U1, V1
        Qk = Q % n
        
        # Process bits of k (skipping the first '1')
        bits = bin(k)[3:]
        
        for bit in bits:
            # Double step
            U = (U * V) % n
            V = (V * V - 2 * Qk) % n
            Qk = (Qk * Qk) % n
            
            if bit == '1':
                # Add step
                U, V = div2_mod(P * U + V, n), div2_mod(D * U + P * V, n)
                Qk = (Qk * Q) % n
        
        return U == 0
    
    # ------------------------------------------------------------------------
    # 3.5 Baillie-PSW Test
    # ------------------------------------------------------------------------
    
    def baillie_psw(self, n: int) -> Tuple[Result, float, str]:
        """
        Baillie-PSW primality test.
        
        Algebraic basis: Combination of Miller-Rabin base 2 and Lucas test.
        No known composite numbers pass both tests.
        """
        start_time = time.perf_counter()
        
        # Base cases
        if n < 2:
            return Result.COMPOSITE, 0.0, "Numbers less than 2 are composite"
        if n == 2 or n == 3:
            return Result.PRIME, 1.0, f"{n} is prime"
        if n % 2 == 0:
            return Result.COMPOSITE, 0.0, "Even numbers > 2 are composite"
        
        # Check small divisors
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                execution_time = time.perf_counter() - start_time
                return Result.PRIME, 1.0, f"{n} is a small prime"
            if n % p == 0:
                execution_time = time.perf_counter() - start_time
                return Result.COMPOSITE, 1.0, f"Divisible by {p}"
        
        # Check perfect squares
        if self.nt.is_perfect_square(n):
            execution_time = time.perf_counter() - start_time
            return Result.COMPOSITE, 1.0, f"{n} is a perfect square"
        
        # Miller-Rabin base 2
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        
        x = self.nt.mod_pow(2, d, n)
        if x == 1 or x == n - 1:
            mr_pass = True
        else:
            mr_pass = False
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    mr_pass = True
                    break
        
        if not mr_pass:
            execution_time = time.perf_counter() - start_time
            return Result.COMPOSITE, 1.0, "Failed Miller-Rabin base 2 test"
        
        # Find D for Lucas test
        D = 5
        found = False
        for _ in range(20):  # Limit attempts
            try:
                j = self.nt.jacobi_symbol(D, n)
                if j == 0:
                    execution_time = time.perf_counter() - start_time
                    return Result.COMPOSITE, 1.0, f"Jacobi symbol ({D}/{n}) = 0"
                if j == -1:
                    found = True
                    break
            except:
                break
            
            if D > 0:
                D = -D - 2
            else:
                D = -D + 2
        
        if not found:
            execution_time = time.perf_counter() - start_time
            return Result.COMPOSITE, 1.0, "Could not find suitable D for Lucas test"
        
        # Lucas test
        if self._lucas_sequence_mod(n, D):
            execution_time = time.perf_counter() - start_time
            # No known counterexamples, but we can't say 100% certain
            return Result.PROBABLY_PRIME, 0.999999, "Passed both Miller-Rabin base 2 and Lucas test (no known counterexamples)"
        else:
            execution_time = time.perf_counter() - start_time
            return Result.COMPOSITE, 1.0, "Failed Lucas probable prime test"
    
    # ------------------------------------------------------------------------
    # 3.6 Lucas-Lehmer Test (for Mersenne numbers)
    # ------------------------------------------------------------------------
    
    def lucas_lehmer(self, p: int) -> Tuple[Result, float, str]:
        """
        Lucas-Lehmer test for Mersenne numbers 2^p - 1.
        
        Algebraic basis: Properties of Lucas sequences modulo Mersenne numbers.
        Deterministic test.
        """
        start_time = time.perf_counter()
        
        if p < 2:
            return Result.COMPOSITE, 0.0, "Exponent must be ≥ 2"
        
        # Check if p is prime (simple trial division)
        if p == 2:
            pass  # 2 is prime
        elif p % 2 == 0:
            execution_time = time.perf_counter() - start_time
            return Result.COMPOSITE, 1.0, f"Exponent {p} is even (not prime)"
        else:
            for i in range(3, int(math.sqrt(p)) + 1, 2):
                if p % i == 0:
                    execution_time = time.perf_counter() - start_time
                    return Result.COMPOSITE, 1.0, f"Exponent {p} is composite"
        
        mersenne = (1 << p) - 1  # 2^p - 1
        s = 4
        
        for i in range(p - 2):
            s = (s * s - 2) % mersenne
        
        execution_time = time.perf_counter() - start_time
        if s == 0:
            return Result.PRIME, 1.0, f"M_{p} = 2^{p}-1 = {mersenne} is prime"
        else:
            return Result.COMPOSITE, 1.0, f"M_{p} = 2^{p}-1 = {mersenne} is composite"
    
    # ------------------------------------------------------------------------
    # 3.7 Simple Deterministic Check (for comparison)
    # ------------------------------------------------------------------------
    
        return True

    # ------------------------------------------------------------------------
    # 3.8 Trial Division Test
    # ------------------------------------------------------------------------
    
    def trial_division(self, n: int) -> Tuple[Result, float, str]:
        """
        Trial division primality test.
        Deterministic for all n.
        """
        start_time = time.perf_counter()
        
        if n < 2:
            return Result.COMPOSITE, 1.0, "Numbers less than 2 are composite"
        if n == 2 or n == 3:
            return Result.PRIME, 1.0, f"{n} is prime"
        if n % 2 == 0:
            return Result.COMPOSITE, 1.0, f"Even number > 2: {n} is composite"
        if n % 3 == 0:
            return Result.COMPOSITE, 1.0, f"Divisible by 3: {n} is composite"
        
        i = 5
        # Limit trial division for extremely large numbers to avoid hang
        limit = min(int(math.sqrt(n)), 10**6) 
        
        while i <= limit:
            if n % i == 0:
                return Result.COMPOSITE, 1.0, f"Found factor {i}"
            if n % (i + 2) == 0:
                return Result.COMPOSITE, 1.0, f"Found factor {i+2}"
            i += 6
            
        execution_time = time.perf_counter() - start_time
        if i * i > n:
            return Result.PRIME, 1.0, "Fully checked all possible factors"
        else:
            return Result.PROBABLY_PRIME, 0.5, f"Checked factors up to {limit}. Incomplete trial division."

    # ------------------------------------------------------------------------
    # 3.9 AKS (Agrawal–Kayal–Saxena) Test
    # ------------------------------------------------------------------------

    def aks(self, n: int) -> Tuple[Result, float, str]:
        """
        AKS primality test (Deterministic).
        Note: Extremely slow for large n (O(log^6 n)).
        """
        start_time = time.perf_counter()
        
        if n < 2:
            return Result.COMPOSITE, 1.0, "n < 2"
        if n == 2:
            return Result.PRIME, 1.0, "2 is prime"
        
        # Step 1: Check if n is a perfect power
        if self.nt.is_perfect_power(n):
            return Result.COMPOSITE, 1.0, f"{n} is a perfect power"
            
        # Step 2: Find the smallest r such that order of n modulo r > (log2 n)^2
        log2n = math.log2(n)
        limit = int(log2n**2)
        r = 2
        while True:
            # Check if order of n mod r > limit
            val = n % r
            if val == 0: # n is divisible by r
                if r < n:
                    return Result.COMPOSITE, 1.0, f"Small factor found: {r}"
                else: # r == n
                    return Result.PRIME, 1.0, f"{n} is prime"
            
            # Simple order finding
            found_order = False
            for k in range(1, limit + 1):
                if pow(n, k, r) == 1:
                    found_order = True
                    break
            
            if not found_order:
                break
            r += 1

        # Step 3: Check gcd(a, n) for a <= r
        for a in range(2, min(r, n)):
            g = self.nt.gcd(a, n)
            if 1 < g < n:
                return Result.COMPOSITE, 1.0, f"gcd({a}, {n}) = {g}"

        # Step 4: If n <= r, it's prime
        if n <= r:
            return Result.PRIME, 1.0, f"n <= r ({r})"

        # Step 5: Polynomial check
        phi_r = self.nt.euler_phi(r)
        poly_limit = int(math.sqrt(phi_r) * log2n)
        
        if n > 100000:
            return Result.PROBABLY_PRIME, 0.9, f"n is too large for AKS polynomial step. Passed steps 1-4."

        # Polynomial multiplication mod (x^r - 1, n)
        def poly_mul(p1, p2, r, n):
            res = {}
            for d1, c1 in p1.items():
                for d2, c2 in p2.items():
                    d = (d1 + d2) % r
                    res[d] = (res.get(d, 0) + c1 * c2) % n
            return res

        def poly_pow(base, exp, r, n):
            res = {0: 1}
            while exp > 0:
                if exp % 2 == 1:
                    res = poly_mul(res, base, r, n)
                base = poly_mul(base, base, r, n)
                exp //= 2
            return res

        for a in range(1, poly_limit + 1):
            lhs = poly_pow({1: 1, 0: a}, n, r, n)
            rhs = {n % r: 1, 0: a}
            
            for d in range(r):
                if lhs.get(d, 0) != rhs.get(d, 0):
                    return Result.COMPOSITE, 1.0, f"Failed polynomial congruence for a={a}"

        return Result.PRIME, 1.0, "Passed all AKS steps"

# ============================================================================
# 4. TEST MANAGER AND ANALYSIS
# ============================================================================

class PrimalityTestManager:
    """Manages all primality tests and provides unified interface."""
    
    def __init__(self):
        self.tester = PrimalityTests()
        self.default_iterations = {
            TestType.SOLOVAY_STRASSEN: 10,
            TestType.MILLER_RABIN: 10,
            TestType.FERMAT: 10,
            TestType.BAILLIE_PSW: 1,  # Single test (deterministic-like)
            TestType.LUCAS_LEHMER: 1,   # Deterministic
            TestType.TRIAL_DIVISION: 1,
            TestType.AKS: 1
        }
    
    def run_test(self, test_type: TestType, n: int, iterations: Optional[int] = None) -> TestResult:
        """
        Run a single primality test.
        
        Args:
            test_type: Type of test to run
            n: Number to test
            iterations: Number of iterations (for probabilistic tests)
            
        Returns:
            TestResult object with all details
        """
        if iterations is None:
            iterations = self.default_iterations.get(test_type, 1)
        
        start_time = time.perf_counter()
        
        try:
            # Handle both Enum and string inputs for robustness
            test_val = test_type.value if hasattr(test_type, 'value') else test_type
            
            if test_val == TestType.SOLOVAY_STRASSEN.value:
                result, certainty, message = self.tester.solovay_strassen(n, iterations)
            elif test_val == TestType.MILLER_RABIN.value:
                result, certainty, message = self.tester.miller_rabin(n, iterations)
            elif test_val == TestType.FERMAT.value:
                result, certainty, message = self.tester.fermat(n, iterations)
            elif test_val == TestType.BAILLIE_PSW.value:
                result, certainty, message = self.tester.baillie_psw(n)
            elif test_val == TestType.LUCAS_LEHMER.value:
                # Check if number is of form 2^p - 1
                p = self._is_mersenne_exponent(n)
                if p is None:
                    execution_time = time.perf_counter() - start_time
                    return TestResult(
                        test_type=test_type,
                        result=Result.ERROR,
                        certainty=0.0,
                        execution_time=execution_time,
                        iterations=0,
                        message=f"{n} is not a Mersenne number (2^p - 1)"
                    )
                result, certainty, message = self.tester.lucas_lehmer(p)
            elif test_val == TestType.TRIAL_DIVISION.value:
                result, certainty, message = self.tester.trial_division(n)
            elif test_val == TestType.AKS.value:
                result, certainty, message = self.tester.aks(n)
            else:
                execution_time = time.perf_counter() - start_time
                return TestResult(
                    test_type=test_type,
                    result=Result.ERROR,
                    certainty=0.0,
                    execution_time=execution_time,
                    iterations=0,
                    message=f"Unknown test type: {test_val}"
                )
            
            execution_time = time.perf_counter() - start_time
            return TestResult(
                test_type=test_type,
                result=result,
                certainty=certainty,
                execution_time=execution_time,
                iterations=iterations,
                message=message
            )
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return TestResult(
                test_type=test_type,
                result=Result.ERROR,
                certainty=0.0,
                execution_time=execution_time,
                iterations=iterations,
                message=f"Error during test: {str(e)}"
            )
    
    def _is_mersenne_exponent(self, n: int) -> Optional[int]:
        """Check if n is of the form 2^p - 1, return p if true."""
        n_plus_1 = n + 1
        if n_plus_1 <= 0:
            return None
        
        # Check if n+1 is a power of 2
        if (n_plus_1 & (n_plus_1 - 1)) != 0:
            return None
        
        # Find the exponent
        p = 0
        temp = n_plus_1
        while temp > 1:
            temp >>= 1
            p += 1
        
        return p
    
    def analyze_number(self, n: int, tests: Optional[List[TestType]] = None) -> NumberAnalysis:
        """
        Run multiple tests on a number and provide comprehensive analysis.
        
        Args:
            n: Number to analyze
            tests: List of tests to run (None = all tests)
            
        Returns:
            NumberAnalysis object with all results
        """
        start_time = time.perf_counter()
        
        if tests is None:
            tests = list(TestType)
        
        analysis = NumberAnalysis(number=n)
        analysis.test_results = []
        
        # Determine if it's a Mersenne number
        is_mersenne = self._is_mersenne_exponent(n) is not None
        
        for test_type in tests:
            # Skip Lucas-Lehmer if not a Mersenne number (unless explicitly requested)
            if test_type == TestType.LUCAS_LEHMER and not is_mersenne:
                continue
            
            result = self.run_test(test_type, n)
            analysis.test_results.append(result)
        
        # Determine overall verdict
        verdicts = [r.result for r in analysis.test_results]
        
        if Result.ERROR in verdicts:
            analysis.overall_verdict = "Analysis incomplete due to errors"
        elif all(r.result == Result.PRIME for r in analysis.test_results):
            analysis.overall_verdict = "Definitely Prime"
            analysis.is_prime = True
        elif all(r.result == Result.COMPOSITE for r in analysis.test_results):
            analysis.overall_verdict = "Definitely Composite"
            analysis.is_prime = False
        elif Result.PROBABLY_PRIME in verdicts and Result.COMPOSITE not in verdicts:
            # Check certainty levels
            max_certainty = max(r.certainty for r in analysis.test_results)
            if max_certainty > 0.9999:
                analysis.overall_verdict = "Almost Certainly Prime"
                analysis.is_prime = True
            else:
                analysis.overall_verdict = "Probably Prime"
                analysis.is_prime = None  # Uncertain
        else:
            # Mixed results
            analysis.overall_verdict = "Inconclusive (conflicting results)"
            analysis.is_prime = None
        
        analysis.total_time = time.perf_counter() - start_time
        return analysis

# ============================================================================
# 5. COMMAND LINE INTERFACE
# ============================================================================

def print_test_result(result: TestResult):
    """Print a single test result in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Test: {result.test_type.value}")
    print(f"Result: {result.result.value}")
    print(f"Certainty: {result.certainty:.6f}")
    print(f"Time: {result.execution_time:.6f} seconds")
    if result.iterations > 1:
        print(f"Iterations: {result.iterations}")
    print(f"Message: {result.message}")
    print(f"{'='*60}")

def print_analysis(analysis: NumberAnalysis):
    """Print complete analysis in a formatted way."""
    print(f"\n{'='*80}")
    print(f"PRIMALITY ANALYSIS FOR n = {analysis.number}")
    print(f"{'='*80}")
    
    print(f"\nOverall Verdict: {analysis.overall_verdict}")
    if analysis.is_prime is not None:
        print(f"Prime Status: {'PRIME' if analysis.is_prime else 'COMPOSITE'}")
    else:
        print(f"Prime Status: UNCERTAIN")
    print(f"Total Analysis Time: {analysis.total_time:.6f} seconds")
    
    print(f"\n{'='*80}")
    print("DETAILED TEST RESULTS:")
    print(f"{'='*80}")
    
    for result in analysis.test_results:
        print_test_result(result)
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    
    # Create summary table
    print(f"\n{'Test':<25} {'Result':<15} {'Certainty':<12} {'Time (s)':<10}")
    print(f"{'-'*25} {'-'*15} {'-'*12} {'-'*10}")
    
    for result in analysis.test_results:
        print(f"{result.test_type.value:<25} "
              f"{result.result.value:<15} "
              f"{result.certainty:<12.6f} "
              f"{result.execution_time:<10.6f}")

def interactive_mode():
    """Run in interactive command-line mode."""
    print("\n" + "="*80)
    print("PRIMALITY TESTING SYSTEM - INTERACTIVE MODE")
    print("="*80)
    
    manager = PrimalityTestManager()
    
    while True:
        print("\n" + "-"*80)
        print("Options:")
        print("  1. Test a single number with all tests")
        print("  2. Compare specific tests on a number")
        print("  3. Test a range of numbers")
        print("  4. Batch test from file")
        print("  5. View test descriptions")
        print("  6. Exit")
        print("-"*80)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            try:
                n = int(input("Enter number to test: "))
                analysis = manager.analyze_number(n)
                print_analysis(analysis)
                
                # Ask if user wants to save results
                save = input("\nSave results to JSON file? (y/n): ").lower()
                if save == 'y':
                    filename = f"primality_{n}.json"
                    with open(filename, 'w') as f:
                        f.write(analysis.to_json())
                    print(f"Results saved to {filename}")
                    
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            try:
                n = int(input("Enter number to test: "))
                
                print("\nAvailable tests:")
                for i, test_type in enumerate(TestType, 1):
                    print(f"  {i}. {test_type.value}")
                
                test_nums = input("\nEnter test numbers (comma-separated, e.g., 1,2,4): ").strip()
                test_indices = [int(x.strip()) - 1 for x in test_nums.split(',')]
                
                selected_tests = []
                for idx in test_indices:
                    if 0 <= idx < len(TestType):
                        selected_tests.append(list(TestType)[idx])
                
                if not selected_tests:
                    print("No valid tests selected.")
                    continue
                
                analysis = NumberAnalysis(number=n)
                analysis.test_results = []
                
                for test_type in selected_tests:
                    result = manager.run_test(test_type, n)
                    analysis.test_results.append(result)
                    print_test_result(result)
                
            except ValueError:
                print("Invalid input.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            try:
                start = int(input("Enter start of range: "))
                end = int(input("Enter end of range: "))
                
                if end < start:
                    print("End must be greater than start.")
                    continue
                if end - start > 100:
                    print("Range too large. Please limit to 100 numbers.")
                    continue
                
                results = []
                for n in range(start, end + 1):
                    print(f"\nTesting {n}...", end="", flush=True)
                    analysis = manager.analyze_number(n)
                    results.append(analysis)
                    print(f" {analysis.overall_verdict}")
                
                # Summary
                primes = [a for a in results if a.is_prime == True]
                composites = [a for a in results if a.is_prime == False]
                uncertain = [a for a in results if a.is_prime is None]
                
                print(f"\n{'='*60}")
                print(f"RANGE SUMMARY: {start} to {end}")
                print(f"{'='*60}")
                print(f"Primes: {len(primes)}")
                print(f"Composites: {len(composites)}")
                print(f"Uncertain: {len(uncertain)}")
                
                if primes:
                    print(f"\nPrime numbers found: {[a.number for a in primes]}")
                
            except ValueError:
                print("Invalid input.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "4":
            filename = input("Enter filename with numbers (one per line): ").strip()
            try:
                with open(filename, 'r') as f:
                    numbers = [int(line.strip()) for line in f if line.strip()]
                
                print(f"\nTesting {len(numbers)} numbers from {filename}...")
                
                results = []
                for i, n in enumerate(numbers, 1):
                    print(f"\rProgress: {i}/{len(numbers)}", end="", flush=True)
                    analysis = manager.analyze_number(n)
                    results.append(analysis)
                
                print(f"\n\nAnalysis complete!")
                
                # Save all results
                output_file = f"batch_results_{int(time.time())}.json"
                with open(output_file, 'w') as f:
                    json.dump([a.to_dict() for a in results], f, indent=2)
                
                print(f"Results saved to {output_file}")
                
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "5":
            print("\n" + "="*80)
            print("TEST DESCRIPTIONS:")
            print("="*80)
            
            tests_info = [
                ("Solovay-Strassen", 
                 "Based on Euler's criterion and quadratic residues. "
                 "Error probability ≤ 2^(-k) where k is iterations."),
                
                ("Miller-Rabin", 
                 "Based on properties of square roots of 1 in finite fields. "
                 "Stronger than Solovay-Strassen. Error probability ≤ 4^(-k)."),
                
                ("Fermat", 
                 "Simple test based on Fermat's Little Theorem. "
                 "Fast but unreliable for Carmichael numbers."),
                
                ("Baillie-PSW", 
                 "Combines Miller-Rabin base 2 and Lucas probable prime test. "
                 "No known composite numbers pass both tests."),
                
                ("Lucas-Lehmer", 
                 "Specialized test for Mersenne numbers (2^p - 1). "
                 "Deterministic and very efficient for this special case."),
                
                ("Trial Division",
                 "Classic method of checking divisibility by all integers up to sqrt(n). "
                 "Deterministic but slow for large n."),
                
                ("AKS",
                 "Agrawal–Kayal–Saxena primality test. First provably deterministic "
                 "polynomial-time primality test. Extremely slow in practice.")
            ]
            
            for name, desc in tests_info:
                print(f"\n{name}:")
                print(f"  {desc}")
        
        elif choice == "6":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")

# ============================================================================
# 6. WEB INTERFACE READY FUNCTIONS
# ============================================================================

def analyze_number_api(n: int, tests: Optional[List[str]] = None) -> Dict:
    """
    API-friendly function to analyze a number.
    
    Args:
        n: Number to test
        tests: List of test names to run (e.g., ['solovay_strassen', 'miller_rabin'])
        
    Returns:
        Dictionary with analysis results
    """
    manager = PrimalityTestManager()
    
    # Convert string test names to TestType enum
    test_types = []
    if tests:
        for test_name in tests:
            try:
                # Convert string to enum (handle various formats)
                test_name_normalized = test_name.upper().replace('-', '_').replace(' ', '_')
                test_type = TestType[test_name_normalized]
                test_types.append(test_type)
            except KeyError:
                # If not found, try to match by value
                for tt in TestType:
                    if test_name.lower() in tt.value.lower():
                        test_types.append(tt)
                        break
    
    analysis = manager.analyze_number(n, test_types or None)
    return analysis.to_dict()

def test_single_api(test_name: str, n: int, iterations: Optional[int] = None) -> Dict:
    """
    API-friendly function to run a single test.
    
    Args:
        test_name: Name of test to run
        n: Number to test
        iterations: Number of iterations
        
    Returns:
        Dictionary with test result
    """
    manager = PrimalityTestManager()
    
    # Convert test name to enum
    try:
        test_name_normalized = test_name.upper().replace('-', '_').replace(' ', '_')
        test_type = TestType[test_name_normalized]
    except KeyError:
        # Try to match by value
        for tt in TestType:
            if test_name.lower() in tt.value.lower():
                test_type = tt
                break
        else:
            return {
                'error': f'Unknown test type: {test_name}',
                'available_tests': [tt.value for tt in TestType]
            }
    
    result = manager.run_test(test_type, n, iterations)
    return result.to_dict()

# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Primality Testing System")
    parser.add_argument("number", nargs="?", type=int, help="Number to test")
    parser.add_argument("--tests", type=str, help="Comma-separated list of tests to run")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations for probabilistic tests")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--api", action="store_true", help="Output API-friendly JSON")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.number is not None:
        manager = PrimalityTestManager()
        
        # Parse tests if provided
        test_types = None
        if args.tests:
            test_names = [t.strip() for t in args.tests.split(',')]
            test_types = []
            for name in test_names:
                try:
                    # Try to match enum name
                    test_name_normalized = name.upper().replace('-', '_').replace(' ', '_')
                    test_types.append(TestType[test_name_normalized])
                except KeyError:
                    # Try to match by value
                    found = False
                    for tt in TestType:
                        if name.lower() in tt.value.lower():
                            test_types.append(tt)
                            found = True
                            break
                    if not found:
                        print(f"Warning: Unknown test '{name}', skipping.")
        
        analysis = manager.analyze_number(args.number, test_types)
        
        if args.json or args.api:
            print(analysis.to_json())
        else:
            print_analysis(analysis)
    else:
        # No arguments, show help and run interactive mode
        parser.print_help()
        print("\n" + "="*80)
        run_interactive = input("\nNo arguments provided. Run interactive mode? (y/n): ").lower()
        if run_interactive == 'y':
            interactive_mode()
        else:
            print("\nExamples:")
            print("  python main.py 1000003")
            print("  python main.py 1000003 --tests miller-rabin,baillie-psw --json")
            print("  python main.py --interactive")