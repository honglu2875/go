#include <pybind11/pybind11.h>

#include "config.h"
#include "FastState.h"
#include "FastBoard.h"
#include "Random.h"
#include "Zobrist.h"


namespace py = pybind11;

PYBIND11_MODULE(go, m) {
py::class_<FastBoard> fast_board(m, "FastBoard");

fast_board.def(py::init<>())
.def("get_boardsize", &FastBoard::get_boardsize)
.def("get_state", &FastBoard::get_state)
.def("get_state_vt", &FastBoard::get_state_vt)
.def("get_vertex", &FastBoard::get_vertex)
.def("set_state", &FastBoard::set_state)
.def("set_state_vt", &FastBoard::set_state_vt)
.def("get_xy", &FastBoard::get_xy)
.def("is_suicide", &FastBoard::is_suicide)
.def("count_pliberties", &FastBoard::count_pliberties)
.def("is_eye", &FastBoard::is_eye)
.def("area_score", &FastBoard::area_score)
.def("get_prisoners", &FastBoard::get_prisoners)
.def("black_to_move", &FastBoard::black_to_move)
.def("white_to_move", &FastBoard::white_to_move)
.def("get_to_move", &FastBoard::get_to_move)
.def("set_to_move", &FastBoard::set_to_move)
.def("move_to_text", &FastBoard::move_to_text)
.def("text_to_move", &FastBoard::text_to_move)
.def("get_stone_list", &FastBoard::get_stone_list)
.def("get_string", &FastBoard::get_string)
.def("reset_board", &FastBoard::reset_board)
.def("display_board", &FastBoard::display_board)
.def("starpoint", &FastBoard::starpoint)
.def("starpoint_vt", &FastBoard::starpoint_vt)
.def_readonly_static("NUM_VERTICES", &FastBoard::NUM_VERTICES)
.def_readonly_static("NO_VERTEX", &FastBoard::NO_VERTEX)
.def_readonly_static("PASS", &FastBoard::PASS)
.def_readonly_static("RESIGN", &FastBoard::RESIGN);

py::enum_<FastBoard::vertex_t>(fast_board, "vertex_t")
.value("BLACK", FastBoard::vertex_t::BLACK)
.value("WHITE", FastBoard::vertex_t::WHITE)
.value("EMPTY", FastBoard::vertex_t::EMPTY)
.value("INVAL", FastBoard::vertex_t::INVAL)
.export_values();

py::class_ <FastState> fast_state(m, "FastState");

fast_state.def (py::init<>())
.def("init_game", &FastState::init_game)
.def("reset_game", &FastState::reset_game)
.def("reset_board", &FastState::reset_board)
.def("play_move", &FastState::play_move)
.def("is_move_legal", &FastState::is_move_legal)
.def("set_komi", &FastState::set_komi)
.def("get_komi", &FastState::get_komi)
.def("set_handicap", &FastState::set_handicap)
.def("get_handicap", &FastState::get_handicap)
.def("get_passes", &FastState::get_passes)
.def("get_to_move", &FastState::get_to_move)
.def("set_to_move", &FastState::set_to_move)
.def("set_passes", &FastState::set_passes)
.def("increment_passes", &FastState::increment_passes)
.def("final_score", &FastState::final_score)
.def("get_symmetry_hash", &FastState::get_symmetry_hash)
.def("get_last_move", &FastState::get_last_move)
.def("display_state", &FastState::display_state)
.def("move_to_text", &FastState::move_to_text)
.def_readwrite("board", &FastState::board)
.def_readwrite("m_komi", &FastState::m_komi)
.def_readwrite("m_handicap", &FastState::m_handicap)
.def_readwrite("m_passes", &FastState::m_passes)
.def_readwrite("m_komove", &FastState::m_komove)
.def_readwrite("m_movenum", &FastState::m_movenum)
.def_readwrite("m_lastmove", &FastState::m_lastmove);

py::class_ <Random> random(m, "Random");

random.def (py::init<>())
.def("seedrandom", &Random::seedrandom)
.def("randuint64", &Random::randuint64);

py::class_ <Zobrist> zobrist(m, "Zobrist");

zobrist.def (py::init<>())
.def_readonly_static("zobrist_empty", &Zobrist::zobrist_empty)
.def_readonly_static("zobrist_blacktomove", &Zobrist::zobrist_blacktomove)
.def_readonly_static("zobrist", &Zobrist::zobrist)
.def_readonly_static("zobrist_ko", &Zobrist::zobrist_ko)
.def_readonly_static("zobrist_pris", &Zobrist::zobrist_pris)
.def_readonly_static("zobrist_pass", &Zobrist::zobrist_pass)
.def("init_zobrist", &Zobrist::init_zobrist);
}
