#include "../utils.h"
#include <map>
#include <string>
#include <cstdint>

using namespace std;

void test_json_parse() {
    string simple = "{\"a\": 1, \"b\": 2}"; 
    map<string, int32_t> m = json_parse(simple);
    assert(m["a"] == 1);
    assert(m["b"] == 2);
}

int main() {
    test_json_parse();
    return 0;
}
