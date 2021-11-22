#include "translation/utils/utils.h"
#include <gtest/gtest.h>

using namespace translation;
using namespace translation::utils;

TEST(MurphiTemplateTests, IndentLines) {
  std::string testString = "line one\nline two\nline three";

  std::string indentedString = indentAllLines(testString);

  std::string expstr = "    line one\n    line two\n    line three";

  ASSERT_STREQ(expstr.c_str(), indentedString.c_str());
}

TEST(MurphiTemplateTests, IndentLinesWithTrailingEndl) {
  std::string testString = "line one \n line two \n line three \n";

  std::string indentedString = indentAllLines(testString);

  // additional \n is removed
  std::string expstr = "    line one \n     line two \n     line three ";

  ASSERT_STREQ(expstr.c_str(), indentedString.c_str());
}

TEST(MurphiTemplateTests, InterleaveFunctions) {
  std::vector<std::string> elems{"hello", "world", "is", "a", "sentence"};
  auto interleaved = interleave<std::string>(elems, " ");
  ASSERT_STREQ(interleaved.c_str(), "hello world is a sentence");
}
