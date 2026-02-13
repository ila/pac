SELECT pac_sum(hash(orders.o_custkey), l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem JOIN part ON lineitem.l_partkey = part.p_partkey 
              JOIN orders ON lineitem.l_orderkey = orders.o_orderkey
WHERE (part.p_brand = 'Brand#12'
        AND part.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        AND lineitem.l_quantity >= 1
        AND lineitem.l_quantity <= 1 + 10
        AND part.p_size BETWEEN 1 AND 5
        AND lineitem.l_shipmode IN ('AIR', 'AIR REG')
        AND lineitem.l_shipinstruct = 'DELIVER IN PERSON') OR 
      (part.p_brand = 'Brand#23'
        AND part.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND lineitem.l_quantity >= 10
        AND lineitem.l_quantity <= 10 + 10
        AND part.p_size BETWEEN 1 AND 10
        AND lineitem.l_shipmode IN ('AIR', 'AIR REG')
        AND lineitem.l_shipinstruct = 'DELIVER IN PERSON') OR 
      (part.p_brand = 'Brand#34'
        AND part.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND lineitem.l_quantity >= 20
        AND lineitem.l_quantity <= 20 + 10
        AND part.p_size BETWEEN 1 AND 15
        AND lineitem.l_shipmode IN ('AIR', 'AIR REG')
        AND lineitem.l_shipinstruct = 'DELIVER IN PERSON');
