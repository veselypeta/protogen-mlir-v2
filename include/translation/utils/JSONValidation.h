#pragma once
#include "inja/inja.hpp"
#include "rapidjson/document.h"
#include <valijson/adapters/rapidjson_adapter.hpp>
#include <valijson/schema.hpp>
#include <valijson/schema_parser.hpp>
#include <valijson/utils/rapidjson_utils.hpp>
#include <valijson/validator.hpp>

namespace JSONValidation {
bool validate_json_doc(const std::string &schema_path,
                       const rapidjson::Document &json_doc);
bool validate_json_str(const std::string &schema_path,
                       const std::string &json_str);
bool validate_json(const std::string &schema_path, const nlohmann::json &json);
bool validate_json_doc_ref(const std::string &schema_path,
                           const std::string &json_doc_path);
} // namespace JSONValidation
