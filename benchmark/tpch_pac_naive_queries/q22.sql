WITH samples AS (
    -- 128 sample positions per customer (privacy unit = customer.c_custkey)
    SELECT c.c_custkey AS pu_key, s.sample_id
    FROM customer c
    CROSS JOIN generate_series(1, 128) AS s(sample_id)
),
sampled_customers AS (
    -- sampled customers: join the customer table with the per-customer sample positions
    SELECT c.*, sm.sample_id
    FROM customer c
    JOIN samples sm ON sm.pu_key = c.c_custkey
),
per_sample AS (
    -- aggregate per sample: compute number of customers and total acctbal by country code
    SELECT
        sm.sample_id,
        substring(sm.c_phone FROM 1 FOR 2) AS cntrycode,
        COUNT(*) AS cust_count,
        SUM(sm.c_acctbal) AS sum_acctbal
    FROM sampled_customers sm
    WHERE
        substring(sm.c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND sm.c_acctbal > (
            SELECT avg(sc2.c_acctbal)
            FROM sampled_customers sc2
            WHERE sc2.c_acctbal > 0.00
              AND substring(sc2.c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        )
        AND NOT EXISTS (
            SELECT 1 FROM orders WHERE o_custkey = sm.c_custkey
        )
    GROUP BY sm.sample_id, substring(sm.c_phone FROM 1 FOR 2)
)
SELECT
    cntrycode,
    pac_aggregate(array_agg(cust_count ORDER BY sample_id), array_agg(cust_count ORDER BY sample_id), 1.0/128, 3) AS numcust,
    pac_aggregate(array_agg(sum_acctbal ORDER BY sample_id), array_agg(cust_count ORDER BY sample_id), 1.0/128, 3) AS totacctbal
FROM per_sample
GROUP BY cntrycode
ORDER BY cntrycode;
