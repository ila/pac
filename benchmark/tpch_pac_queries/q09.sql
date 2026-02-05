SELECT n_name AS nation, extract(year FROM o_orderdate) AS o_year,
       pac_sum(hash(orders.o_custkey), l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity) AS sum_profit
FROM part JOIN lineitem ON part.p_partkey = lineitem.l_partkey
          JOIN supplier ON supplier.s_suppkey = lineitem.l_suppkey
          JOIN partsupp ON partsupp.ps_partkey = part.p_partkey AND partsupp.ps_suppkey = supplier.s_suppkey
          JOIN orders ON orders.o_orderkey = lineitem.l_orderkey
          JOIN nation ON nation.n_nationkey = supplier.s_nationkey
WHERE part.p_name LIKE '%green%'
GROUP BY n_name, extract(year FROM o_orderdate)
ORDER BY n_name, o_year DESC;
