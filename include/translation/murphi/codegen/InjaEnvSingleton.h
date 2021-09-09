#pragma once
#include <inja/inja.hpp>

class InjaEnvSingleton {
public:
  static inja::Environment &getInstance();
  InjaEnvSingleton() = delete;
  InjaEnvSingleton(InjaEnvSingleton const &) = delete;
  void operator=(InjaEnvSingleton const &) = delete;

protected:
  static inja::Environment *env;

private:
  ~InjaEnvSingleton();
};
