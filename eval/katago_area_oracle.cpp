// Qualification adapter only. Linked against the pinned, unmodified KataGo source.
// No neural network or GoZero scoring implementation participates in this oracle.
#include "game/board.h"
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  Board::initHash();
  int size;
  std::string flat;
  while (std::cin >> size >> flat) {
    if (size < 1 || size > Board::MAX_LEN || flat.size() != size*size)
      throw std::runtime_error("Invalid oracle input");
    std::string diagram;
    for (int y=0; y<size; y++) diagram += flat.substr(y*size,size) + "\n";
    Board board = Board::parseBoard(size,size,diagram);
    Color owner[Board::MAX_ARR_SIZE];
    board.calculateArea(owner,true,true,true,true);
    for (int y=0; y<size; y++) for (int x=0; x<size; x++) {
      Loc p=Location::getLoc(x,y,size);
      char actual = board.colors[p]==P_BLACK ? 'X' : board.colors[p]==P_WHITE ? 'O' : '.';
      if (actual != flat[y*size+x]) throw std::runtime_error("Oracle setup altered stones");
      std::cout << (int)owner[p];
    }
    std::cout << '\n';
  }
}
