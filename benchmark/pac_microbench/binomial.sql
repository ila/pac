WITH bit_counts AS (SELECT bit_count(hash(c_custkey)) AS ones FROM customer),
       observed AS (SELECT ones, COUNT(*) AS observed_count FROM bit_counts GROUP BY ones),
       expected AS ( -- Expected binomial distribution: C(64,k) * 0.5^64 
                    SELECT ones, observed_count,
                           (SELECT COUNT(*) FROM customer) * -- Binomial probability * total rows    
                           (SELECT EXP(  -- log(C(64,k)) + 64*log(0.5)
                                         -- log(C(64,k)) = sum(log(65-i) - log(i)) for i=1..k  
                                       CASE WHEN ones = 0 THEN 0 
                                       ELSE (SELECT SUM(LN(65.0 - i) - LN(i)) FROM range(1, ones + 1) t(i))  
                                       END+ 64 * LN(0.5))) AS expected_count FROM observed)
SELECT ones AS num_ones, observed_count,  
       ROUND(expected_count, 1) AS expected_count,   
       ROUND(100.0 * observed_count / SUM(observed_count) OVER (), 2) AS observed_pct,  
       ROUND(100.0 * expected_count / SUM(expected_count) OVER (), 2) AS expected_pct,  
       ROUND((observed_count - expected_count) / SQRT(expected_count), 2) AS z_score   
FROM expected    
WHERE ones BETWEEN 20 AND 44 -- Focus on likely range (mean ± 3 std dev)
ORDER BY ones;   
  
SELECT AVG(bit_count(hash(c_custkey))) AS mean_ones,   STDDEV(bit_count(hash(c_custkey))) AS stddev_ones, 32.0 AS expected_mean, 4.0 AS expected_stddev
FROM customer;   -- Summary statistics  
