SELECT pac_sum(hash(orders.o_custkey), l_extendedprice * l_discount) AS revenue
FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
