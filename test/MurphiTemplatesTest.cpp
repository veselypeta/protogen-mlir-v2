#include "translation/utils/utils.h"
#include <gtest/gtest.h>

using namespace translation;
using namespace translation::utils;

TEST(MurphiTemplateTests, IndentLines) {
  std::string testString = "line one\nline two\nline three\n";

  std::string indentedString = indentAllLines(testString);

  std::string expstr = "\tline one\n\tline two\n\tline three\n";

  ASSERT_STREQ(expstr.c_str(), indentedString.c_str());
}

TEST(MurphiTemplateTests, IndentLinesWithoutTrailingEndl) {
  std::string testString = "line one \n line two \n line three";

  std::string indentedString = indentAllLines(testString);

  // If the line does not include a newline - a newline will be added
  std::string expstr = "\tline one \n\t line two \n\t line three\n";

  ASSERT_STREQ(expstr.c_str(), indentedString.c_str());
}

TEST(MurphiTemplateTests, InterleaveFunctions) {
  std::vector<std::string> elems{"hello", "world", "is", "a", "sentence"};
  auto interleaved = interleave<std::string>(elems, " ");
  ASSERT_STREQ(interleaved.c_str(), "hello world is a sentence");
}
