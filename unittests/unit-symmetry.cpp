// Unit tests for the symmetry class

// Let Catch provide main():
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <string>
#include "symmetry.h"

TEST_CASE("Point Group Properties", "[symmetry]") {
  SECTION("D2h") {
    symmetry sym((std::string) "d2h");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 8);

    // Ensure diagonal of product table is Ag
    for (int i = 1; i < 9; i++) {
      REQUIRE(sym.getProduct(i, i) == 1);
    }

    // A few assorted products
    REQUIRE(sym.getProduct(4, 6) == 7);  // B1g * B2g = B3g
    REQUIRE(sym.getProduct(4, 3) == 2);  // B1g * B2u = B3u
    REQUIRE(sym.getProduct(5, 3) == 7);  // B1u * B2u = B3g
  }

  SECTION("C2v") {
    symmetry sym((std::string) "c2v");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 4);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }

  SECTION("C2h") {
    symmetry sym((std::string) "c2h");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 4);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }

  SECTION("D2") {
    symmetry sym((std::string) "d2");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 4);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }

  SECTION("Cs") {
    symmetry sym((std::string) "cs");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 2);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }

  SECTION("C2") {
    symmetry sym((std::string) "c2");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 2);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }

  SECTION("Ci") {
    symmetry sym((std::string) "ci");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 2);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }

  SECTION("C1") {
    symmetry sym((std::string) "c1");

    // Ensure the product_table is the correct size
    REQUIRE(sym.product_table.rows() == 1);
    REQUIRE(sym.product_table.rows() == sym.product_table.cols());
  }
}

// TODO
// TEST_CASE("Symmetry Functions", "[functions]") {
//   SECTION("getSymmetry") {
//
//   }
// }
