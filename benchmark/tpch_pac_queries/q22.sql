SELECT cntrycode,
       pac_count(hash(c_custkey)) AS numcust, 
       pac_sum(hash(c_custkey), c_acctbal) AS totacctbal
FROM (SELECT substring(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal, c_custkey
      FROM customer
      WHERE substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND pac_filter_gt(
              c_acctbal,
              (SELECT pac_avg_counters(hash(c_custkey), c_acctbal) 
                 FROM customer
                WHERE c_acctbal > 0.00
                  AND substring(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')))
        AND NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = customer.c_custkey)) AS custsale
GROUP BY cntrycode
ORDER BY cntrycode;
