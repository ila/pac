//
// Test PAC Parser - JSON serialization and metadata management
//
#include "include/test_pac_parser.hpp"
#include "../include/pac_parser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace duckdb {

#define TEST_ASSERT(condition, message)                                                                                \
	do {                                                                                                               \
		if (!(condition)) {                                                                                            \
			std::cerr << "FAILED: " << message << std::endl;                                                           \
			std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl;                                          \
			throw std::runtime_error(message);                                                                         \
		}                                                                                                              \
	} while (0)

void TestPACParser::TestJSONSerialization() {
	// Create test metadata
	PACTableMetadata metadata("test_table");
	metadata.primary_key_columns = {"id", "tenant_id"};
	metadata.links.push_back(PACLink("user_id", "users", "id"));
	metadata.links.push_back(PACLink("org_id", "organizations", "id"));
	metadata.protected_columns = {"salary", "ssn", "email"};
	// Serialize to JSON
	string json = PACMetadataManager::Get().SerializeToJSON(metadata);
	// Verify JSON contains expected fields
	TEST_ASSERT(json.find("\"table_name\": \"test_table\"") != string::npos, "JSON should contain table_name");
	TEST_ASSERT(json.find("\"id\"") != string::npos, "JSON should contain id");
	TEST_ASSERT(json.find("\"tenant_id\"") != string::npos, "JSON should contain tenant_id");
	TEST_ASSERT(json.find("\"user_id\"") != string::npos, "JSON should contain user_id");
	TEST_ASSERT(json.find("\"users\"") != string::npos, "JSON should contain users");
	TEST_ASSERT(json.find("\"salary\"") != string::npos, "JSON should contain salary");
	// Deserialize from JSON
	PACTableMetadata deserialized = PACMetadataManager::Get().DeserializeFromJSON(json);
	// Verify deserialized data matches original
	TEST_ASSERT(deserialized.table_name == "test_table", "Deserialized table_name should match");
	TEST_ASSERT(deserialized.primary_key_columns.size() == 2, "Deserialized should have 2 PK columns");
	TEST_ASSERT(deserialized.primary_key_columns[0] == "id", "First PK should be id");
	TEST_ASSERT(deserialized.primary_key_columns[1] == "tenant_id", "Second PK should be tenant_id");
	TEST_ASSERT(deserialized.links.size() == 2, "Deserialized should have 2 links");
	TEST_ASSERT(deserialized.links[0].local_columns.size() == 1, "First link should have 1 local column");
	TEST_ASSERT(deserialized.links[0].local_columns[0] == "user_id", "First link local column should be user_id");
	TEST_ASSERT(deserialized.links[0].referenced_table == "users", "First link referenced table should be users");
	TEST_ASSERT(deserialized.links[0].referenced_columns.size() == 1, "First link should have 1 referenced column");
	TEST_ASSERT(deserialized.links[0].referenced_columns[0] == "id", "First link referenced column should be id");
	TEST_ASSERT(deserialized.protected_columns.size() == 3, "Deserialized should have 3 protected columns");
	TEST_ASSERT(deserialized.protected_columns[0] == "salary", "First protected column should be salary");
}

void TestPACParser::TestMetadataManager() {
	auto &manager = PACMetadataManager::Get();
	manager.Clear();
	// Test adding metadata
	PACTableMetadata metadata1("table1");
	metadata1.primary_key_columns = {"id"};
	manager.AddOrUpdateTable("table1", metadata1);
	// Test getting metadata
	TEST_ASSERT(manager.HasMetadata("table1"), "Manager should have table1");
	auto *retrieved = manager.GetTableMetadata("table1");
	TEST_ASSERT(retrieved != nullptr, "Retrieved metadata should not be null");
	TEST_ASSERT(retrieved->table_name == "table1", "Retrieved table name should be table1");
	TEST_ASSERT(retrieved->primary_key_columns.size() == 1, "Retrieved should have 1 PK column");
	// Test non-existent table
	TEST_ASSERT(!manager.HasMetadata("nonexistent"), "Manager should not have nonexistent table");
	TEST_ASSERT(manager.GetTableMetadata("nonexistent") == nullptr, "Nonexistent table should return null");
	// Test adding another table
	PACTableMetadata metadata2("table2");
	metadata2.protected_columns = {"col1", "col2"};
	manager.AddOrUpdateTable("table2", metadata2);
	TEST_ASSERT(manager.HasMetadata("table2"), "Manager should have table2");
	// Test updating existing table
	metadata1.links.push_back(PACLink("fk", "table2", "id"));
	manager.AddOrUpdateTable("table1", metadata1);
	retrieved = manager.GetTableMetadata("table1");
	TEST_ASSERT(retrieved->links.size() == 1, "Updated table should have 1 link");
	// Test clear
	manager.Clear();
	TEST_ASSERT(!manager.HasMetadata("table1"), "Manager should not have table1 after clear");
	TEST_ASSERT(!manager.HasMetadata("table2"), "Manager should not have table2 after clear");
}

void TestPACParser::TestFilePersistence() {
	auto &manager = PACMetadataManager::Get();
	manager.Clear();
	// Create test metadata
	PACTableMetadata metadata1("users");
	metadata1.primary_key_columns = {"user_id"};
	metadata1.protected_columns = {"email", "password"};
	manager.AddOrUpdateTable("users", metadata1);
	PACTableMetadata metadata2("orders");
	metadata2.primary_key_columns = {"order_id"};
	metadata2.links.push_back(PACLink("user_id", "users", "user_id"));
	manager.AddOrUpdateTable("orders", metadata2);
	// Save to file
	string filepath = "/tmp/test_pac_metadata_testdb_main.json";
	manager.SaveToFile(filepath);
	// Verify file exists and contains expected content
	std::ifstream file(filepath);
	TEST_ASSERT(file.is_open(), "File should be open");
	std::stringstream buffer;
	buffer << file.rdbuf();
	file.close();
	string content = buffer.str();
	TEST_ASSERT(content.find("\"users\"") != string::npos, "File content should contain users");
	TEST_ASSERT(content.find("\"orders\"") != string::npos, "File content should contain orders");
	TEST_ASSERT(content.find("\"email\"") != string::npos, "File content should contain email");
	// Clear and reload
	manager.Clear();
	TEST_ASSERT(!manager.HasMetadata("users"), "Manager should not have users after clear");
	manager.LoadFromFile(filepath);
	// Verify data was loaded correctly
	TEST_ASSERT(manager.HasMetadata("users"), "Manager should have users after load");
	TEST_ASSERT(manager.HasMetadata("orders"), "Manager should have orders after load");
	auto *users = manager.GetTableMetadata("users");
	TEST_ASSERT(users != nullptr, "Users metadata should not be null");
	TEST_ASSERT(users->primary_key_columns.size() == 1, "Users should have 1 PK column");
	TEST_ASSERT(users->protected_columns.size() == 2, "Users should have 2 protected columns");
	auto *orders = manager.GetTableMetadata("orders");
	TEST_ASSERT(orders != nullptr, "Orders metadata should not be null");
	TEST_ASSERT(orders->links.size() == 1, "Orders should have 1 link");
	TEST_ASSERT(orders->links[0].referenced_table == "users", "Orders link should reference users");
	// Cleanup
	std::remove(filepath.c_str());
	manager.Clear();
}

void TestPACParser::TestCreatePACTableParsing() {
	// Test basic CREATE PAC TABLE
	string sql1 = "CREATE PAC TABLE users (id INTEGER, name VARCHAR, PAC KEY (id))";
	PACTableMetadata metadata1;
	string stripped1;
	bool result1 = PACParserExtension::ParseCreatePACTable(sql1, stripped1, metadata1);
	TEST_ASSERT(result1, "Parse should succeed");
	TEST_ASSERT(metadata1.table_name == "users", "Table name should be users");
	TEST_ASSERT(metadata1.primary_key_columns.size() == 1, "Should have 1 PK column");
	TEST_ASSERT(metadata1.primary_key_columns[0] == "id", "PK column should be id");
	TEST_ASSERT(stripped1.find("CREATE TABLE") != string::npos, "Stripped SQL should contain CREATE TABLE");
	TEST_ASSERT(stripped1.find("PAC KEY") == string::npos, "Stripped SQL should not contain PAC KEY");
	// Test with PAC LINK
	string sql2 = "CREATE PAC TABLE orders (id INTEGER, user_id INTEGER, PAC LINK (user_id) REFERENCES users(id))";
	PACTableMetadata metadata2;
	string stripped2;
	bool result2 = PACParserExtension::ParseCreatePACTable(sql2, stripped2, metadata2);
	TEST_ASSERT(result2, "Parse should succeed");
	TEST_ASSERT(metadata2.table_name == "orders", "Table name should be orders");
	TEST_ASSERT(metadata2.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata2.links[0].local_columns.size() == 1, "Link should have 1 local column");
	TEST_ASSERT(metadata2.links[0].local_columns[0] == "user_id", "Link local column should be user_id");
	TEST_ASSERT(metadata2.links[0].referenced_table == "users", "Link referenced table should be users");
	TEST_ASSERT(metadata2.links[0].referenced_columns.size() == 1, "Link should have 1 referenced column");
	TEST_ASSERT(metadata2.links[0].referenced_columns[0] == "id", "Link referenced column should be id");
	// Test with PROTECTED
	string sql3 = "CREATE PAC TABLE sensitive (id INTEGER, ssn VARCHAR, PROTECTED (ssn))";
	PACTableMetadata metadata3;
	string stripped3;
	bool result3 = PACParserExtension::ParseCreatePACTable(sql3, stripped3, metadata3);
	TEST_ASSERT(result3, "Parse should succeed");
	TEST_ASSERT(metadata3.table_name == "sensitive", "Table name should be sensitive");
	TEST_ASSERT(metadata3.protected_columns.size() == 1, "Should have 1 protected column");
	TEST_ASSERT(metadata3.protected_columns[0] == "ssn", "Protected column should be ssn");
	// Test with multiple clauses
	string sql4 = "CREATE PAC TABLE employees (emp_id INTEGER, dept_id INTEGER, salary INTEGER, "
	              "PAC KEY (emp_id), PAC LINK (dept_id) REFERENCES departments(id), "
	              "PROTECTED (salary))";
	PACTableMetadata metadata4;
	string stripped4;
	bool result4 = PACParserExtension::ParseCreatePACTable(sql4, stripped4, metadata4);
	TEST_ASSERT(result4, "Parse should succeed");
	TEST_ASSERT(metadata4.table_name == "employees", "Table name should be employees");
	TEST_ASSERT(metadata4.primary_key_columns.size() == 1, "Should have 1 PK column");
	TEST_ASSERT(metadata4.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata4.protected_columns.size() == 1, "Should have 1 protected column");
}

void TestPACParser::TestAlterTablePACParsing() {
	auto &manager = PACMetadataManager::Get();
	manager.Clear();
	// First create a table with initial metadata
	PACTableMetadata initial("products");
	initial.primary_key_columns = {"product_id"};
	manager.AddOrUpdateTable("products", initial);
	// Test ALTER PAC TABLE ADD PAC LINK
	string sql1 = "ALTER PAC TABLE products ADD PAC LINK (category_id) REFERENCES categories(id)";
	PACTableMetadata metadata1;
	string stripped1;
	bool result1 = PACParserExtension::ParseAlterTableAddPAC(sql1, stripped1, metadata1);
	TEST_ASSERT(result1, "Parse should succeed");
	TEST_ASSERT(metadata1.table_name == "products", "Table name should be products");
	TEST_ASSERT(metadata1.primary_key_columns.size() == 1, "Should preserve existing PK");
	TEST_ASSERT(metadata1.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata1.links[0].local_columns.size() == 1, "Link should have 1 local column");
	TEST_ASSERT(metadata1.links[0].local_columns[0] == "category_id", "Link local column should be category_id");
	// Test ALTER PAC TABLE ADD PROTECTED
	string sql2 = "ALTER PAC TABLE products ADD PROTECTED (price, cost)";
	PACTableMetadata metadata2;
	string stripped2;
	bool result2 = PACParserExtension::ParseAlterTableAddPAC(sql2, stripped2, metadata2);
	TEST_ASSERT(result2, "Parse should succeed");
	TEST_ASSERT(metadata2.table_name == "products", "Table name should be products");
	TEST_ASSERT(metadata2.protected_columns.size() == 2, "Should have 2 protected columns");
	manager.Clear();
}

void TestPACParser::TestCompositeKeyParsing() {
	auto &manager = PACMetadataManager::Get();
	manager.Clear();

	// Test single-column FK (should still work)
	string sql1 = "ALTER PAC TABLE orders ADD PAC LINK (customer_id) REFERENCES customers(id)";
	PACTableMetadata metadata1;
	string stripped1;
	bool result1 = PACParserExtension::ParseAlterTableAddPAC(sql1, stripped1, metadata1);
	TEST_ASSERT(result1, "Single-column FK parse should succeed");
	TEST_ASSERT(metadata1.table_name == "orders", "Table name should be orders");
	TEST_ASSERT(metadata1.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata1.links[0].local_columns.size() == 1, "Link should have 1 local column");
	TEST_ASSERT(metadata1.links[0].local_columns[0] == "customer_id", "Link local column should be customer_id");
	TEST_ASSERT(metadata1.links[0].referenced_table == "customers", "Referenced table should be customers");
	TEST_ASSERT(metadata1.links[0].referenced_columns.size() == 1, "Link should have 1 referenced column");
	TEST_ASSERT(metadata1.links[0].referenced_columns[0] == "id", "Referenced column should be id");

	// Test composite FK (two columns)
	string sql2 =
	    "ALTER PAC TABLE lineitem ADD PAC LINK (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)";
	PACTableMetadata metadata2;
	string stripped2;
	bool result2 = PACParserExtension::ParseAlterTableAddPAC(sql2, stripped2, metadata2);
	TEST_ASSERT(result2, "Composite FK parse should succeed");
	TEST_ASSERT(metadata2.table_name == "lineitem", "Table name should be lineitem");
	TEST_ASSERT(metadata2.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata2.links[0].local_columns.size() == 2, "Link should have 2 local columns");
	TEST_ASSERT(metadata2.links[0].local_columns[0] == "l_partkey", "First local column should be l_partkey");
	TEST_ASSERT(metadata2.links[0].local_columns[1] == "l_suppkey", "Second local column should be l_suppkey");
	TEST_ASSERT(metadata2.links[0].referenced_table == "partsupp", "Referenced table should be partsupp");
	TEST_ASSERT(metadata2.links[0].referenced_columns.size() == 2, "Link should have 2 referenced columns");
	TEST_ASSERT(metadata2.links[0].referenced_columns[0] == "ps_partkey",
	            "First referenced column should be ps_partkey");
	TEST_ASSERT(metadata2.links[0].referenced_columns[1] == "ps_suppkey",
	            "Second referenced column should be ps_suppkey");

	// Test composite FK with spaces (three columns)
	string sql3 = "ALTER PAC TABLE complex ADD PAC LINK (col1, col2, col3) REFERENCES target(ref1, ref2, ref3)";
	PACTableMetadata metadata3;
	string stripped3;
	bool result3 = PACParserExtension::ParseAlterTableAddPAC(sql3, stripped3, metadata3);
	TEST_ASSERT(result3, "Three-column composite FK parse should succeed");
	TEST_ASSERT(metadata3.table_name == "complex", "Table name should be complex");
	TEST_ASSERT(metadata3.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata3.links[0].local_columns.size() == 3, "Link should have 3 local columns");
	TEST_ASSERT(metadata3.links[0].referenced_columns.size() == 3, "Link should have 3 referenced columns");
	TEST_ASSERT(metadata3.links[0].local_columns[0] == "col1", "First local column should be col1");
	TEST_ASSERT(metadata3.links[0].local_columns[1] == "col2", "Second local column should be col2");
	TEST_ASSERT(metadata3.links[0].local_columns[2] == "col3", "Third local column should be col3");
	TEST_ASSERT(metadata3.links[0].referenced_columns[0] == "ref1", "First referenced column should be ref1");
	TEST_ASSERT(metadata3.links[0].referenced_columns[1] == "ref2", "Second referenced column should be ref2");
	TEST_ASSERT(metadata3.links[0].referenced_columns[2] == "ref3", "Third referenced column should be ref3");

	// Test CREATE PAC TABLE with composite FK
	string sql4 = "CREATE PAC TABLE items (item_id INTEGER, part_id INTEGER, supplier_id INTEGER, "
	              "PAC LINK (part_id, supplier_id) REFERENCES partsupplier(ps_part, ps_supplier))";
	PACTableMetadata metadata4;
	string stripped4;
	bool result4 = PACParserExtension::ParseCreatePACTable(sql4, stripped4, metadata4);
	TEST_ASSERT(result4, "CREATE PAC TABLE with composite FK should succeed");
	TEST_ASSERT(metadata4.table_name == "items", "Table name should be items");
	TEST_ASSERT(metadata4.links.size() == 1, "Should have 1 link");
	TEST_ASSERT(metadata4.links[0].local_columns.size() == 2, "Link should have 2 local columns");
	TEST_ASSERT(metadata4.links[0].referenced_columns.size() == 2, "Link should have 2 referenced columns");
	TEST_ASSERT(metadata4.links[0].local_columns[0] == "part_id", "First local column should be part_id");
	TEST_ASSERT(metadata4.links[0].local_columns[1] == "supplier_id", "Second local column should be supplier_id");

	// Test serialization and deserialization of composite keys
	PACTableMetadata serialize_test("test_composite");
	serialize_test.primary_key_columns = {"id"};
	serialize_test.links.push_back(PACLink(vector<string> {"col1", "col2"}, "target", vector<string> {"ref1", "ref2"}));
	serialize_test.protected_columns = {"sensitive"};

	string json = PACMetadataManager::Get().SerializeToJSON(serialize_test);
	TEST_ASSERT(json.find("\"local_columns\"") != string::npos, "JSON should contain local_columns array");
	TEST_ASSERT(json.find("\"referenced_columns\"") != string::npos, "JSON should contain referenced_columns array");
	TEST_ASSERT(json.find("\"col1\"") != string::npos, "JSON should contain col1");
	TEST_ASSERT(json.find("\"col2\"") != string::npos, "JSON should contain col2");
	TEST_ASSERT(json.find("\"ref1\"") != string::npos, "JSON should contain ref1");
	TEST_ASSERT(json.find("\"ref2\"") != string::npos, "JSON should contain ref2");

	PACTableMetadata deserialized = PACMetadataManager::Get().DeserializeFromJSON(json);
	TEST_ASSERT(deserialized.table_name == "test_composite", "Deserialized table name should match");
	TEST_ASSERT(deserialized.links.size() == 1, "Deserialized should have 1 link");
	TEST_ASSERT(deserialized.links[0].local_columns.size() == 2, "Deserialized link should have 2 local columns");
	TEST_ASSERT(deserialized.links[0].referenced_columns.size() == 2,
	            "Deserialized link should have 2 referenced columns");
	TEST_ASSERT(deserialized.links[0].local_columns[0] == "col1", "First deserialized local column should be col1");
	TEST_ASSERT(deserialized.links[0].local_columns[1] == "col2", "Second deserialized local column should be col2");
	TEST_ASSERT(deserialized.links[0].referenced_columns[0] == "ref1",
	            "First deserialized referenced column should be ref1");
	TEST_ASSERT(deserialized.links[0].referenced_columns[1] == "ref2",
	            "Second deserialized referenced column should be ref2");

	manager.Clear();
}

void TestPACParser::TestRegexPatterns() {
	// Test ExtractTableName with various formats
	string sql1 = "CREATE PAC TABLE users (id INTEGER)";
	string table1 = PACParserExtension::ExtractTableName(sql1, true);
	TEST_ASSERT(table1 == "users", "Should extract table name from CREATE PAC TABLE");

	string sql2 = "CREATE TABLE IF NOT EXISTS products (id INTEGER)";
	string table2 = PACParserExtension::ExtractTableName(sql2, true);
	TEST_ASSERT(table2 == "products", "Should extract table name from CREATE TABLE IF NOT EXISTS");

	string sql3 = "ALTER TABLE orders ADD COLUMN status VARCHAR";
	string table3 = PACParserExtension::ExtractTableName(sql3, false);
	TEST_ASSERT(table3 == "orders", "Should extract table name from ALTER TABLE");

	// Test ExtractPACPrimaryKey
	vector<string> pk_cols1;
	bool pk_result1 = PACParserExtension::ExtractPACPrimaryKey("PAC KEY (id)", pk_cols1);
	TEST_ASSERT(pk_result1, "Should extract single PK column");
	TEST_ASSERT(pk_cols1.size() == 1, "Should have 1 PK column");
	TEST_ASSERT(pk_cols1[0] == "id", "PK column should be id");

	vector<string> pk_cols2;
	bool pk_result2 = PACParserExtension::ExtractPACPrimaryKey("PAC KEY (user_id, tenant_id)", pk_cols2);
	TEST_ASSERT(pk_result2, "Should extract composite PK");
	TEST_ASSERT(pk_cols2.size() == 2, "Should have 2 PK columns");
	TEST_ASSERT(pk_cols2[0] == "user_id", "First PK should be user_id");
	TEST_ASSERT(pk_cols2[1] == "tenant_id", "Second PK should be tenant_id");

	// Test ExtractPACLink
	PACLink link1;
	bool link_result1 = PACParserExtension::ExtractPACLink("PAC LINK (user_id) REFERENCES users(id)", link1);
	TEST_ASSERT(link_result1, "Should extract single-column link");
	TEST_ASSERT(link1.local_columns.size() == 1, "Link should have 1 local column");
	TEST_ASSERT(link1.local_columns[0] == "user_id", "Local column should be user_id");
	TEST_ASSERT(link1.referenced_table == "users", "Referenced table should be users");
	TEST_ASSERT(link1.referenced_columns.size() == 1, "Link should have 1 referenced column");
	TEST_ASSERT(link1.referenced_columns[0] == "id", "Referenced column should be id");

	PACLink link2;
	bool link_result2 = PACParserExtension::ExtractPACLink(
	    "PAC LINK (part_id, supplier_id) REFERENCES partsupplier(ps_part, ps_supplier)", link2);
	TEST_ASSERT(link_result2, "Should extract composite link");
	TEST_ASSERT(link2.local_columns.size() == 2, "Link should have 2 local columns");
	TEST_ASSERT(link2.referenced_columns.size() == 2, "Link should have 2 referenced columns");

	// Test ExtractProtectedColumns
	vector<string> protected1;
	bool protected_result1 = PACParserExtension::ExtractProtectedColumns("PROTECTED (salary)", protected1);
	TEST_ASSERT(protected_result1, "Should extract single protected column");
	TEST_ASSERT(protected1.size() == 1, "Should have 1 protected column");
	TEST_ASSERT(protected1[0] == "salary", "Protected column should be salary");

	vector<string> protected2;
	bool protected_result2 = PACParserExtension::ExtractProtectedColumns("PROTECTED (salary, ssn, email)", protected2);
	TEST_ASSERT(protected_result2, "Should extract multiple protected columns");
	TEST_ASSERT(protected2.size() == 3, "Should have 3 protected columns");
	TEST_ASSERT(protected2[0] == "salary", "First protected column should be salary");
	TEST_ASSERT(protected2[1] == "ssn", "Second protected column should be ssn");
	TEST_ASSERT(protected2[2] == "email", "Third protected column should be email");

	// Test StripPACClauses
	string sql_with_pac = "CREATE PAC TABLE test (id INTEGER, user_id INTEGER, salary INTEGER, "
	                      "PAC KEY (id), PAC LINK (user_id) REFERENCES users(id), PROTECTED (salary))";
	string stripped = PACParserExtension::StripPACClauses(sql_with_pac);
	TEST_ASSERT(stripped.find("PAC KEY") == string::npos, "Stripped SQL should not contain PAC KEY");
	TEST_ASSERT(stripped.find("PAC LINK") == string::npos, "Stripped SQL should not contain PAC LINK");
	TEST_ASSERT(stripped.find("PROTECTED") == string::npos, "Stripped SQL should not contain PROTECTED");
	TEST_ASSERT(stripped.find("id INTEGER") != string::npos, "Stripped SQL should still contain column definitions");
}

void TestPACParser::TestDropPACConstraints() {
	auto &manager = PACMetadataManager::Get();
	manager.Clear();

	// Setup: Create metadata with protected columns and links
	PACTableMetadata metadata("drop_test_table");
	metadata.primary_key_columns = {"id"};
	metadata.protected_columns = {"col1", "col2", "col3"};
	metadata.links.push_back(PACLink("ref1", "ref_table", "id"));
	metadata.links.push_back(PACLink(vector<string> {"key1", "key2"}, "comp_ref_table", vector<string> {"id1", "id2"}));
	manager.AddOrUpdateTable("drop_test_table", metadata);

	// Test removing protected column
	auto *current = manager.GetTableMetadata("drop_test_table");
	TEST_ASSERT(current->protected_columns.size() == 3, "Should have 3 protected columns initially");

	// Simulate dropping col1
	PACTableMetadata updated1 = *current;
	updated1.protected_columns.erase(
	    std::remove(updated1.protected_columns.begin(), updated1.protected_columns.end(), "col1"),
	    updated1.protected_columns.end());
	manager.AddOrUpdateTable("drop_test_table", updated1);

	current = manager.GetTableMetadata("drop_test_table");
	TEST_ASSERT(current->protected_columns.size() == 2, "Should have 2 protected columns after drop");
	TEST_ASSERT(std::find(current->protected_columns.begin(), current->protected_columns.end(), "col1") ==
	                current->protected_columns.end(),
	            "col1 should be removed");
	TEST_ASSERT(std::find(current->protected_columns.begin(), current->protected_columns.end(), "col2") !=
	                current->protected_columns.end(),
	            "col2 should still exist");

	// Test removing PAC LINK (single column)
	TEST_ASSERT(current->links.size() == 2, "Should have 2 links initially");

	PACTableMetadata updated2 = *current;
	updated2.links.erase(std::remove_if(updated2.links.begin(), updated2.links.end(),
	                                    [](const PACLink &link) {
		                                    return link.local_columns.size() == 1 && link.local_columns[0] == "ref1";
	                                    }),
	                     updated2.links.end());
	manager.AddOrUpdateTable("drop_test_table", updated2);

	current = manager.GetTableMetadata("drop_test_table");
	TEST_ASSERT(current->links.size() == 1, "Should have 1 link after drop");
	TEST_ASSERT(current->links[0].local_columns.size() == 2, "Remaining link should be composite");

	// Test removing composite PAC LINK
	PACTableMetadata updated3 = *current;
	updated3.links.erase(std::remove_if(updated3.links.begin(), updated3.links.end(),
	                                    [](const PACLink &link) {
		                                    return link.local_columns.size() == 2 && link.local_columns[0] == "key1" &&
		                                           link.local_columns[1] == "key2";
	                                    }),
	                     updated3.links.end());
	manager.AddOrUpdateTable("drop_test_table", updated3);

	current = manager.GetTableMetadata("drop_test_table");
	TEST_ASSERT(current->links.empty(), "Should have no links after dropping all");

	// Test RemoveTable functionality
	TEST_ASSERT(manager.HasMetadata("drop_test_table"), "Table should exist before removal");
	manager.RemoveTable("drop_test_table");
	TEST_ASSERT(!manager.HasMetadata("drop_test_table"), "Table should not exist after removal");
	TEST_ASSERT(manager.GetTableMetadata("drop_test_table") == nullptr, "Should return null after removal");
}

void TestPACParser::TestDropTableCleanup() {
	auto &manager = PACMetadataManager::Get();
	manager.Clear();

	// Setup: Create tables with links
	PACTableMetadata target_table("target_table");
	target_table.primary_key_columns = {"id"};
	manager.AddOrUpdateTable("target_table", target_table);

	PACTableMetadata link_table1("link_table1");
	link_table1.links.push_back(PACLink("target_id", "target_table", "id"));
	manager.AddOrUpdateTable("link_table1", link_table1);

	PACTableMetadata link_table2("link_table2");
	link_table2.links.push_back(PACLink("target_id", "target_table", "id"));
	link_table2.links.push_back(PACLink("other_id", "other_table", "id"));
	manager.AddOrUpdateTable("link_table2", link_table2);

	// Verify initial state
	TEST_ASSERT(manager.HasMetadata("target_table"), "target_table should exist");
	TEST_ASSERT(manager.HasMetadata("link_table1"), "link_table1 should exist");
	TEST_ASSERT(manager.HasMetadata("link_table2"), "link_table2 should exist");

	// Simulate DROP TABLE target_table - need to clean up referencing links
	auto all_tables = manager.GetAllTableNames();
	for (const auto &table_name : all_tables) {
		if (table_name == "target_table") {
			continue;
		}

		auto metadata = manager.GetTableMetadata(table_name);
		if (!metadata) {
			continue;
		}

		// Check if this table has links to target_table
		bool has_link = false;
		for (const auto &link : metadata->links) {
			if (link.referenced_table == "target_table") {
				has_link = true;
				break;
			}
		}

		if (has_link) {
			// Remove the link
			PACTableMetadata updated = *metadata;
			updated.links.erase(
			    std::remove_if(updated.links.begin(), updated.links.end(),
			                   [](const PACLink &link) { return link.referenced_table == "target_table"; }),
			    updated.links.end());
			manager.AddOrUpdateTable(table_name, updated);
		}
	}

	// Remove target_table metadata
	manager.RemoveTable("target_table");

	// Verify cleanup
	TEST_ASSERT(!manager.HasMetadata("target_table"), "target_table should be removed");

	auto *link1 = manager.GetTableMetadata("link_table1");
	TEST_ASSERT(link1 != nullptr, "link_table1 should still exist");
	TEST_ASSERT(link1->links.empty(), "link_table1 should have no links after cleanup");

	auto *link2 = manager.GetTableMetadata("link_table2");
	TEST_ASSERT(link2 != nullptr, "link_table2 should still exist");
	TEST_ASSERT(link2->links.size() == 1, "link_table2 should have 1 link after cleanup");
	TEST_ASSERT(link2->links[0].referenced_table == "other_table", "Remaining link should reference other_table");

	// Cleanup
	manager.Clear();
}

void TestPACParser::RunAllTests() {
	std::cout << "Running TestJSONSerialization..." << std::endl;
	TestJSONSerialization();
	std::cout << "PASSED: TestJSONSerialization" << std::endl;

	std::cout << "Running TestMetadataManager..." << std::endl;
	TestMetadataManager();
	std::cout << "PASSED: TestMetadataManager" << std::endl;

	std::cout << "Running TestFilePersistence..." << std::endl;
	TestFilePersistence();
	std::cout << "PASSED: TestFilePersistence" << std::endl;

	std::cout << "Running TestCreatePACTableParsing..." << std::endl;
	TestCreatePACTableParsing();
	std::cout << "PASSED: TestCreatePACTableParsing" << std::endl;

	std::cout << "Running TestAlterTablePACParsing..." << std::endl;
	TestAlterTablePACParsing();
	std::cout << "PASSED: TestAlterTablePACParsing" << std::endl;

	std::cout << "Running TestRegexPatterns..." << std::endl;
	TestRegexPatterns();
	std::cout << "PASSED: TestRegexPatterns" << std::endl;

	std::cout << "Running TestDropPACConstraints..." << std::endl;
	TestDropPACConstraints();
	std::cout << "PASSED: TestDropPACConstraints" << std::endl;

	std::cout << "Running TestDropTableCleanup..." << std::endl;
	TestDropTableCleanup();
	std::cout << "PASSED: TestDropTableCleanup" << std::endl;

	std::cout << "\nAll tests passed!" << std::endl;
}

} // namespace duckdb
