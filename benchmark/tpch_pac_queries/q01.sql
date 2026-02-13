SELECT l_returnflag, l_linestatus,
       pac_sum(hash(orders.o_custkey), l_quantity) AS sum_qty,
       pac_sum(hash(orders.o_custkey), l_extendedprice) AS sum_base_price,
       pac_sum( hash(orders.o_custkey), l_extendedprice * (1 - l_discount)) AS sum_disc_price,
       pac_sum(hash(orders.o_custkey), l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
       pac_avg(hash(orders.o_custkey), l_quantity) AS avg_qty,
       pac_avg(hash(orders.o_custkey), l_extendedprice) AS avg_price,
       pac_avg(hash(orders.o_custkey), l_discount) AS avg_disc,
       pac_count(hash(orders.o_custkey), 1) AS count_order
FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
