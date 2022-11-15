#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

int main(int argc, char* argv[])
{
    std::ifstream t("demo.txt");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());
    t.close();
    std::cout << contents << std::endl;
    return EXIT_SUCCESS;
}