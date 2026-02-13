WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
             CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
per_sample AS (
    SELECT
        c_count,
        count(*) AS custdist
    FROM (
        SELECT
            c_custkey,
            count(o_orderkey)
        FROM
            customer
        INNER JOIN samples ON pu_key = c_custkey
        LEFT OUTER JOIN orders ON c_custkey = o_custkey
        AND o_comment NOT LIKE '%special%requests%'
    GROUP BY
        c_custkey) AS c_orders (c_custkey,
            c_count)
    GROUP BY
        c_count
    ORDER BY
        custdist DESC,
        c_count DESC)
SELECT
    c_count,
    pac_aggregate(array_agg(custdist ORDER BY c_count), array_agg(custdist ORDER BY c_count), 1.0/128, 3) AS custdist
FROM per_sample
GROUP BY c_count
ORDER BY c_count DESC;




