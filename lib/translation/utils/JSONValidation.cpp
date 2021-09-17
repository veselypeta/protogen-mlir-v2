#include "JSONValidation.h"

namespace JSONValidation {

namespace {
// Callback functions for fetching remote documents
const rapidjson::Document *fetch_doc(const std::string &uri) {

  auto *fetchedRoot = new rapidjson::Document();

  // load the documents from within the base directory
  std::string doc_path = std::string(schema_base_directory) + uri;
  if(!valijson::utils::loadDocument(doc_path, *fetchedRoot)){
    delete fetchedRoot;
    std::string error_msg = doc_path + " : is an invalid reference to a document";
    throw std::runtime_error(error_msg.c_str());
  }
    return fetchedRoot;
}

void free_doc(const rapidjson::Document *adapter) {
  delete adapter;
}
} // namespace

bool validate_json_doc(const std::string &schema_path,
                       const rapidjson::Document &json_doc) {
  // parse the schema file into a rapijson document
  rapidjson::Document schema_doc;
  if (!valijson::utils::loadDocument(schema_path, schema_doc)) {
    throw std::runtime_error("Failed to load schema document");
  }
  // create the valijson schema
  valijson::Schema schema;
  {
    valijson::SchemaParser schemaParser;
    valijson::adapters::RapidJsonAdapter schemaAdapter(schema_doc);
    schemaParser.populateSchema(schemaAdapter, schema, fetch_doc,
                                free_doc);
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