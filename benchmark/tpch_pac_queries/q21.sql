SELECT
    s_name,
    pac_count(hash(orders.o_custkey)) AS numwait
FROM
    supplier
        JOIN lineitem l1
             ON s_suppkey = l1.l_suppkey
        JOIN orders
             ON o_orderkey = l1.l_orderkey
        JOIN nation
             ON s_nationkey = n_nationkey
WHERE
    o_orderstatus = 'F'
  AND l1.l_receiptdate > l1.l_commitdate
  AND EXISTS (
    SELECT 1
    FROM lineitem l2
             JOIN orders o2
                  ON l2.l_orderkey = o2.o_orderkey
             JOIN customer c2
                  ON o2.o_custkey = c2.c_custkey
    WHERE l2.l_orderkey = l1.l_orderkey
      AND l2.l_suppkey <> l1.l_suppkey
)
  AND NOT EXISTS (
    SELECT 1
    FROM lineitem l3
             JOIN orders o3
                  ON l3.l_orderkey = o3.o_orderkey
             JOIN customer c3
                  ON o3.o_custkey = c3.c_custkey
    WHERE l3.l_orderkey = l1.l_orderkey
      AND l3.l_suppkey <> l1.l_suppkey
      AND l3.l_receiptdate > l3.l_commitdate
)
  AND n_name = 'SAUDI ARABIA'
GROUP BY
    s_name
ORDER BY
    numwait DESC,
    s_name
    LIMIT 100;
