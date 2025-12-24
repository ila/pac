SELECT
    pac_sum(hash(customer.c_custkey), l_extendedprice) / 7.0 AS avg_yearly
FROM
    lineitem
        JOIN part
             ON lineitem.l_partkey = part.p_partkey
        JOIN orders
             ON lineitem.l_orderkey = orders.o_orderkey
        JOIN customer
             ON orders.o_custkey = customer.c_custkey
WHERE
    part.p_brand = 'Brand#23'
  AND part.p_container = 'MED BOX'
  AND lineitem.l_quantity < (
    SELECT 0.2 * AVG(l_quantity)
    FROM lineitem AS l_sub
    WHERE l_sub.l_partkey = part.p_partkey
);
