// Reconstruct pinned KataGo V7 inputs from complete archived games.
// KataGo library copyright/license is retained in KATAGO_LICENSE.md.
#include "game/board.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "external/nlohmann_json/json.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if(argc != 5) throw std::runtime_error("input.jsonl spatial.u8 globals.f32 audit.u8 required");
    Board::initHash();
    std::ifstream input(argv[1]);
    std::ofstream spatial(argv[2],std::ios::binary), globals(argv[3],std::ios::binary), audit(argv[4],std::ios::binary);
    if(!input || !spatial || !globals || !audit) throw std::runtime_error("Failed to open input/output");
    size_t rows=0, games=0;
    std::string line;
    while(std::getline(input,line)) {
      auto item=nlohmann::json::parse(line);
      int n=item.at("size").get<int>();
      if(n < 1 || n > Board::MAX_LEN) throw std::runtime_error("Board size out of range");
      auto actions=item.at("actions").get<std::vector<int>>();
      Board board(n,n);
      Rules rules(Rules::KO_POSITIONAL,Rules::SCORING_AREA,Rules::TAX_NONE,true,false,Rules::WHB_ZERO,false,item.at("komi").get<float>());
      BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes());
      MiscNNInputParams params;
      std::vector<float> bin(n*n*22), glob(19);
      std::vector<unsigned char> packed(bin.size()), state(n*n*2+1);
      Player pla=P_BLACK;
      for(int action: actions) {
        if(action < 0 || action > n*n || hist.isGameFinished) throw std::runtime_error("Invalid action or post-terminal observation");
        NNInputs::fillRowV7(board,hist,pla,params,n,n,true,bin.data(),glob.data());
        for(size_t i=0; i<bin.size(); i++) {
          if(bin[i] != 0.0f && bin[i] != 1.0f) throw std::runtime_error("Nonbinary spatial feature");
          packed[i]=static_cast<unsigned char>(bin[i]);
        }
        for(int i=0; i<n*n; i++) {
          Loc loc=Location::getLoc(i%n,i/n,n);
          state[i]=board.colors[loc];
          state[n*n+i]=hist.isLegal(board,loc,pla);
        }
        state[2*n*n]=hist.isLegal(board,Board::PASS_LOC,pla);
        Loc move=action == n*n ? Board::PASS_LOC : Location::getLoc(action%n,action/n,n);
        if(!hist.isLegal(board,move,pla)) throw std::runtime_error("Archived action illegal under pinned rules");
        spatial.write(reinterpret_cast<const char*>(packed.data()),packed.size());
        globals.write(reinterpret_cast<const char*>(glob.data()),glob.size()*sizeof(float));
        audit.write(reinterpret_cast<const char*>(state.data()),state.size());
        hist.makeBoardMoveAssumeLegal(board,move,pla,nullptr);
        pla=getOpp(pla); rows++;
      }
      games++;
    }
    spatial.close(); globals.close(); audit.close();
    if(!spatial || !globals || !audit) throw std::runtime_error("Failed to write outputs");
    std::cout << nlohmann::json({{"games",games},{"positions",rows},{"status","passed"}}).dump() << std::endl;
  } catch(const std::exception& error) { std::cerr << error.what() << std::endl; return 1; }
}
