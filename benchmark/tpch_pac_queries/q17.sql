SELECT pac_sum(hash(orders.o_custkey), l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem JOIN part ON lineitem.l_partkey = part.p_partkey JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
WHERE part.p_brand = 'Brand#23'
  AND part.p_container = 'MED BOX'
  AND lineitem.l_quantity < (SELECT 0.2 * pac_avg(hash(o_sub.o_custkey), l_quantity)
                             FROM lineitem AS l_sub JOIN orders AS o_sub ON l_sub.l_orderkey = o_sub.o_orderkey
                             WHERE l_sub.l_partkey = part.p_partkey);
