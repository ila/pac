SELECT
    l_returnflag,
    l_linestatus,

    pac_sum(hash(customer.c_custkey), l_quantity) AS sum_qty,

    pac_sum(hash(customer.c_custkey), l_extendedprice) AS sum_base_price,

    pac_sum(
            hash(customer.c_custkey),
            l_extendedprice * (1 - l_discount)
    ) AS sum_disc_price,

    pac_sum(
            hash(customer.c_custkey),
            l_extendedprice * (1 - l_discount) * (1 + l_tax)
    ) AS sum_charge,

    pac_sum(hash(customer.c_custkey), l_quantity)
        / pac_count(hash(customer.c_custkey)) AS avg_qty,

    pac_sum(hash(customer.c_custkey), l_extendedprice)
        / pac_count(hash(customer.c_custkey)) AS avg_price,

    pac_sum(hash(customer.c_custkey), l_discount)
        / pac_count(hash(customer.c_custkey)) AS avg_disc,

    pac_count(hash(customer.c_custkey)) AS count_order

FROM
    lineitem
        JOIN orders
             ON lineitem.l_orderkey = orders.o_orderkey
        JOIN customer
             ON orders.o_custkey = customer.c_custkey

WHERE
    l_shipdate <= DATE '1998-09-02'

GROUP BY
    l_returnflag,
    l_linestatus

ORDER BY
    l_returnflag,
    l_linestatus;
