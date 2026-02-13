SELECT CAST(pac_noised(
              list_transform(
                list_zip(list_transform(
                           pac_sum_counters(hash(l_orderkey), CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END),
                           lambda y: CAST(y as DECIMAL(18,2))),
                         list_transform(
                           pac_sum_counters(hash(l_orderkey), l_extendedprice * (1 - l_discount)), 
                           lambda y: CAST(y as DECIMAL(18,2)))),
                lambda x: CAST(100.0 * (x[1] / x[2]) AS FLOAT))) AS DOUBLE) AS promo_revenue
FROM lineitem JOIN part ON l_partkey = p_partkey 
WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01';
