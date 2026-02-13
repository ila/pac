SELECT 100.0 * pac_sum(hash(orders.o_custkey), CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
             / pac_sum(hash(orders.o_custkey), l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem JOIN part ON l_partkey = p_partkey JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01';
