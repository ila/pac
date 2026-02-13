SELECT o_year, 
       CAST(pac_noised(
              list_transform(
                list_zip(
                  list_transform(
                    pac_sum_counters(hash(all_nations.c_custkey), (CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END)),
                    lambda y: CAST(y AS DECIMAL(18,2))),
                  list_transform(
                    pac_sum_counters(hash(all_nations.c_custkey), volume),
                    lambda y: CAST(y AS DECIMAL(18,2)))),
                lambda x: CAST(x[1] / x[2] AS FLOAT))) AS FLOAT) AS mkt_share
FROM (SELECT EXTRACT(year FROM o_orderdate) AS o_year, l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation, customer.c_custkey
         FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
         WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey AND o_custkey = c_custkey
           AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey AND s_nationkey = n2.n_nationkey
           AND r_name = 'AMERICA'
           AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
           AND p_type = 'ECONOMY ANODIZED STEEL') AS all_nations
GROUP BY o_year
ORDER BY o_year;
