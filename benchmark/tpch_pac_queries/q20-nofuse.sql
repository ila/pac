SELECT s_name, s_address
FROM supplier JOIN nation ON s_nationkey = n_nationkey 
WHERE s_suppkey IN (SELECT ps_suppkey 
                    FROM partsupp
                    WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
                      AND pac_filter(
                            list_transform(
                              list_transform((SELECT pac_sum_counters(hash(orders.o_custkey), l_quantity)
                                                FROM lineitem JOIN orders ON l_orderkey = o_orderkey 
                                               WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey 
                                                 AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'),
                                lambda y: CAST(y AS DECIMAL(18,3))),
                              lambda x: ps_availqty > 0.5 * x))) 
                      AND n_name = 'CANADA'
ORDER BY s_name;
