// Persistent bounded V7 inference replay. See KATAGO_LICENSE.md.
#include "game/board.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "external/nlohmann_json/json.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Json=nlohmann::json;
static constexpr size_t MAX_REQUEST=4*1024*1024;
static constexpr size_t MAX_OUTPUT=64*1024*1024;

static void header(const Json& value) {
  std::string text=value.dump();uint32_t n=static_cast<uint32_t>(text.size());
  unsigned char bytes[4]={static_cast<unsigned char>(n),static_cast<unsigned char>(n>>8),
                         static_cast<unsigned char>(n>>16),static_cast<unsigned char>(n>>24)};
  std::cout.write(reinterpret_cast<const char*>(bytes),4);
  std::cout.write(text.data(),text.size());
}

static void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}

static int integer(const Json& value,int lo,int hi) {
  require(value.is_number_integer(),"Expected an integer");
  int64_t n=value.get<int64_t>();require(n>=lo && n<=hi,"Integer out of range");
  return static_cast<int>(n);
}

int main() {
  try {
    uint32_t endian=1;require(*reinterpret_cast<unsigned char*>(&endian)==1 && sizeof(float)==4,"Requires little-endian float32");
    Board::initHash();
    while(true) {
      unsigned char prefix[4];std::cin.read(reinterpret_cast<char*>(prefix),4);
      if(std::cin.gcount()==0 && std::cin.eof())return 0;
      require(std::cin.gcount()==4,"Truncated request frame");
      uint32_t n=uint32_t(prefix[0])|(uint32_t(prefix[1])<<8)|(uint32_t(prefix[2])<<16)|(uint32_t(prefix[3])<<24);
      require(n>0 && n<=MAX_REQUEST,"Request frame exceeds bound");
      std::string text(n,'\0');std::cin.read(&text[0],n);
      require(static_cast<size_t>(std::cin.gcount())==n,"Truncated request body");
      Json id=nullptr;
      try {
        auto request=Json::parse(text);
        require(request.is_object() && request.size()==6,"Wrong request fields");
        for(const char* key:{"version","id","size","komi","histories","starts"})require(request.contains(key),"Missing request field");
        require(integer(request.at("version"),1,1)==1,"Wrong protocol version");
        id=integer(request.at("id"),1,2147483647);
        int size=integer(request.at("size"),1,Board::MAX_LEN);int area=size*size;
        require(request.at("komi").is_number(),"Invalid komi");
        double komi=request.at("komi").get<double>();require(std::isfinite(komi) && std::abs(komi)<=1000,"Invalid komi");
        require(static_cast<double>(static_cast<float>(komi))==komi,"Komi must be exactly representable as float32");
        const auto& histories=request.at("histories");const auto& starts=request.at("starts");
        require(histories.is_array() && histories.size()>0 && histories.size()<=128,"History batch out of range");
        require(starts.is_array() && starts.size()==histories.size(),"Suffix starts differ");
        size_t rows=0;std::vector<std::vector<int>> tapes;std::vector<int> beginnings,offsets={0};
        for(size_t i=0;i<histories.size();i++) {
          const auto& h=histories.at(i);require(h.is_array() && h.size()<2048,"History exceeds context bound");
          std::vector<int> actions;actions.reserve(h.size());
          for(const auto& action:h)actions.push_back(integer(action,0,area));
          int start=integer(starts.at(i),0,static_cast<int>(actions.size()));
          rows+=actions.size()+1-start;offsets.push_back(static_cast<int>(rows));
          tapes.push_back(std::move(actions));beginnings.push_back(start);
        }
        size_t spatial_bytes=rows*area*22,global_bytes=rows*19*sizeof(float),audit_bytes=rows*(2*area+1);
        require(spatial_bytes+global_bytes+audit_bytes<=MAX_OUTPUT,"Output exceeds bound");
        std::vector<unsigned char> spatial;spatial.reserve(spatial_bytes);
        std::vector<float> globals;globals.reserve(rows*19);
        std::vector<unsigned char> audit;audit.reserve(audit_bytes);
        std::vector<float> bin(area*22),glob(19);
        for(size_t game=0;game<tapes.size();game++) {
          Board board(size,size);
          Rules rules(Rules::KO_POSITIONAL,Rules::SCORING_AREA,Rules::TAX_NONE,true,false,Rules::WHB_ZERO,false,static_cast<float>(komi));
          BoardHistory history(board,P_BLACK,rules,0,BoardHistoryModes());MiscNNInputParams params;
          Player player=P_BLACK;const auto& actions=tapes[game];
          for(size_t t=0;t<=actions.size();t++) {
            require(!history.isGameFinished,"Terminal leaf or post-terminal history");
            if(t>=static_cast<size_t>(beginnings[game])) {
              NNInputs::fillRowV7(board,history,player,params,size,size,true,bin.data(),glob.data());
              for(float x:bin){require(x==0.0f || x==1.0f,"Nonbinary spatial feature");spatial.push_back(static_cast<unsigned char>(x));}
              for(float x:glob){require(std::isfinite(x),"Nonfinite global feature");globals.push_back(x);}
              for(int i=0;i<area;i++)audit.push_back(board.colors[Location::getLoc(i%size,i/size,size)]);
              for(int i=0;i<area;i++)audit.push_back(history.isLegal(board,Location::getLoc(i%size,i/size,size),player));
              audit.push_back(history.isLegal(board,Board::PASS_LOC,player));
            }
            if(t==actions.size())break;
            int action=actions[t];Loc move=action==area?Board::PASS_LOC:Location::getLoc(action%size,action/size,size);
            require(history.isLegal(board,move,player),"Illegal history action");
            history.makeBoardMoveAssumeLegal(board,move,player,nullptr);player=getOpp(player);
          }
        }
        require(spatial.size()==spatial_bytes && globals.size()*4==global_bytes && audit.size()==audit_bytes,"Output shape mismatch");
        header(Json{{"version",1},{"id",id},{"status","ok"},{"size",size},{"rows",rows},{"offsets",offsets},
                    {"spatial_bytes",spatial_bytes},{"global_bytes",global_bytes},{"audit_bytes",audit_bytes}});
        std::cout.write(reinterpret_cast<const char*>(spatial.data()),spatial.size());
        std::cout.write(reinterpret_cast<const char*>(globals.data()),global_bytes);
        std::cout.write(reinterpret_cast<const char*>(audit.data()),audit.size());
      } catch(const std::exception& error) {
        header(Json{{"version",1},{"id",id},{"status","error"},{"error",error.what()}});
      }
      std::cout.flush();require(bool(std::cout),"Failed response write");
    }
  } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}
}
