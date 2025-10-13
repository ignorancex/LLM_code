#include "fieldquantity.hpp"

#include <stdexcept>
#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include "system.hpp"
#include "world.hpp"

Grid FieldQuantity::grid() const {
  return system()->grid();
}

void FieldQuantity::addToField(Field& f) const {
  if (!sameFieldDimensions(*this, f))
    throw std::invalid_argument(
        "Can not add the quantity to given field because the fields are "
        "incompatible.");
  f += *this;  // += checks assuredZero before calling eval()
}

std::vector<real> FieldQuantity::average() const {
  return fieldAverage(eval());
}

Field FieldQuantity::getRGB() const {
  return fieldGetRGB(eval());
}

const World* FieldQuantity::world() const {
  const System* sys = system().get();
  if (sys)
    return sys->world();
  return nullptr;
}
