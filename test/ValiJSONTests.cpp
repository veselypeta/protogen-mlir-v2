#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>

TEST(ValiJSON, basic_json_test) {
  nlohmann::json j = {{"productId", 21}};
  ASSERT_TRUE(
      JSONValidation::validate_json("../../test/valijson_test_schema.json", j));
}

TEST(ValijJSON, basic_doc_test) {
  ASSERT_TRUE(JSONValidation::validate_json_doc_ref(
      "../../test/valijson_test_schema.json",
      "../../test/valijson_test_doc.json"));
}

TEST(ValiJSON, basic_json_str) {
  std::string json_str = "{\"productId\", 21}";
  ASSERT_TRUE(JSONValidation::validate_json_str(
      "../../test/valijson_test_schema.json", json_str));
}