-- q14: 64-possible-worlds-semantics with lambdas. 
-- we calculate the expression (100*x[1]/x[2]) in a list_transform-lambda for all 64 outcomes
-- the outcome of that expression is then reduced to a single noised value with pac_noised() and cast it to its expected type
-- since that expression contains two aggregates, we list_zip two counters into a list of two values first
-- the inner list-_transforms are just to cast the DOUBLE counter values back to their original type
-- the final computed expression is then reduced to a single noised double value using pac_noised and cast to the exptected type
SELECT CAST(pac_noised(
              list_transform(
                list_zip(list_transform(
                           pac_sum_counters(hash(o_custkey), CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END),
                           lambda y: CAST(y as DECIMAL(18,2))),
                         list_transform(
                           pac_sum_counters(hash(o_custkey), l_extendedprice * (1 - l_discount)), 
                           lambda y: CAST(y as DECIMAL(18,2)))),
                lambda x: CAST(100.0 * (x[1] / x[2]) AS FLOAT))) AS DOUBLE) AS promo_revenue
FROM lineitem JOIN part ON l_partkey = p_partkey JOIN orders ON l_orderkey = o_orderkey
WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01';
