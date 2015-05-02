#include <iostream>
#include <string>
#include <vector>
#include <locale>
#include <iterator>
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"

int main(int argc, char** argv)
{
  boost::program_options::options_description description("Options");

  description.add_options()
    ("help,h","Print program help message")
    ("test-path,t","The path to the test set")
    ("classifier-path,-c","The path to the file that contains the classifier"
     " data")
    ("algorithm,a","The training algorithm used");

  boost::program_options::variables_map vm;

  boost::program_options::store(
      boost::program_options::parse_command_line(argc, argv, description),
      vm);

  if (vm.count("help") || argc <=1)
  {
    std::cout << "Usage : " << argv[0] << std::endl <<
      description << std::endl;
    return 0;
  }

  // std::string hOption1("--help");
  // std::string hOption2("-h");
  // for (int i = 0; i < argc; i++)
  // {
    // std::string argument(argv[i]);
    // if (hOption1.compare(argument) == 0 ||
        // hOption2.compare(argument) == 0)

    // {
      // std::cout << "Usage : " << argv[0] << "
      // return 0;
    // }
  // }
  return 0;
}
