#include "JSONValidation.h"

namespace JSONValidation {
bool validate_json_doc(const std::string &schema_path,
                       const rapidjson::Document &json_doc) {
  // parse the schema file into a rapijson document
  rapidjson::Document schemaDoc;
  if (!valijson::utils::loadDocument(schema_path, schemaDoc)) {
    throw std::runtime_error("Failed to load schema document");
  }
  // create the valijson schema
  valijson::Schema schema;
  {
    valijson::SchemaParser schemaParser;
    valijson::adapters::RapidJsonAdapter schemaAdapter;
    schemaParser.populateSchema(schemaAdapter, schema);
  }

  // run the validation
  valijson::Validator validator;
  valijson::adapters::RapidJsonAdapter valAdapter(json_doc);

  return validator.validate(schema, valAdapter, nullptr);
}

bool validate_json_str(const std::string &schema_path,
                       const std::string &json_str) {
  rapidjson::Document json_doc;
  json_doc.Parse(json_str.c_str());
  return validate_json_doc(schema_path, json_doc);
}

bool validate_json(const std::string &schema_path, const nlohmann::json &json) {
  return validate_json_str(schema_path, nlohmann::to_string(json));
}

bool validate_json_doc_ref(const std::string &schema_path,
                           const std::string &json_doc_path) {
  rapidjson::Document json_doc;
  if (!valijson::utils::loadDocument(json_doc_path, json_doc)) {
    throw std::runtime_error("Failed to load target document");
  }
  return validate_json_doc(schema_path, json_doc);
}
} // namespace JSONValidation