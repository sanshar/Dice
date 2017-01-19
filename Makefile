CXX = mpiicpc
CC = mpiicpc
FLAGS = -ipo -xcore-avx2 -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I./boost/

LFLAGS = -L./boost/lib -lboost_serialization -lboost_mpi

SRC_cisd = CISD.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_hci = HCI.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp HCInonessentials.cpp
SRC_forcyrus = forCyrus.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp HCInonessentials.cpp
SRC_hci2 = HCI.cpp HCIbasics2.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_Excitations = Excitations.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp


BOOST_SERIALIZATION_OBJ =       boost/lib/serialization/src/basic_archive.o\
                                boost/lib/serialization/src/extended_type_info.o\
                                boost/lib/serialization/src/basic_iarchive.o\
                                boost/lib/serialization/src/extended_type_info_no_rtti.o\
                                boost/lib/serialization/src/basic_iserializer.o\
                                boost/lib/serialization/src/extended_type_info_typeid.o\
                                boost/lib/serialization/src/basic_oarchive.o\
                                boost/lib/serialization/src/polymorphic_iarchive.o\
                                boost/lib/serialization/src/basic_oserializer.o\
                                boost/lib/serialization/src/polymorphic_oarchive.o\
                                boost/lib/serialization/src/basic_pointer_iserializer.o\
                                boost/lib/serialization/src/stl_port.o\
                                boost/lib/serialization/src/basic_pointer_oserializer.o\
                                boost/lib/serialization/src/text_iarchive.o\
                                boost/lib/serialization/src/basic_serializer_map.o\
                                boost/lib/serialization/src/text_oarchive.o\
                                boost/lib/serialization/src/basic_text_iprimitive.o\
                                boost/lib/serialization/src/text_wiarchive.o\
                                boost/lib/serialization/src/basic_text_oprimitive.o\
                                boost/lib/serialization/src/text_woarchive.o\
                                boost/lib/serialization/src/basic_text_wiprimitive.o\
                                boost/lib/serialization/src/utf8_codecvt_facet.o\
                                boost/lib/serialization/src/basic_text_woprimitive.o\
                                boost/lib/serialization/src/void_cast.o\
                                boost/lib/serialization/src/basic_xml_archive.o\
                                boost/lib/serialization/src/xml_grammar.o\
                                boost/lib/serialization/src/binary_iarchive.o\
                                boost/lib/serialization/src/xml_iarchive.o\
                                boost/lib/serialization/src/binary_oarchive.o\
                                boost/lib/serialization/src/xml_oarchive.o\
                                boost/lib/serialization/src/xml_wgrammar.o\
                                boost/lib/serialization/src/xml_wiarchive.o\
                                boost/lib/serialization/src/codecvt_null.o\
                                boost/lib/serialization/src/xml_woarchive.o\
                                boost/lib/serialization/src/binary_wiarchive.o\
                                boost/lib/serialization/src/binary_woarchive.o

BOOST_MPI_OBJ   =               boost/lib/mpi/src/broadcast.o\
                                boost/lib/mpi/src/environment.o\
                                boost/lib/mpi/src/intercommunicator.o\
                                boost/lib/mpi/src/packed_oarchive.o\
                                boost/lib/mpi/src/request.o\
                                boost/lib/mpi/src/communicator.o\
                                boost/lib/mpi/src/exception.o\
                                boost/lib/mpi/src/mpi_datatype_cache.o\
                                boost/lib/mpi/src/packed_skeleton_iarchive.o\
                                boost/lib/mpi/src/text_skeleton_oarchive.o\
                                boost/lib/mpi/src/computation_tree.o\
                                boost/lib/mpi/src/graph_communicator.o\
                                boost/lib/mpi/src/mpi_datatype_oarchive.o\
                                boost/lib/mpi/src/packed_skeleton_oarchive.o\
                                boost/lib/mpi/src/timer.o\
                                boost/lib/mpi/src/content_oarchive.o\
                                boost/lib/mpi/src/group.o\
                                boost/lib/mpi/src/packed_iarchive.o\
                                boost/lib/mpi/src/point_to_point.o


OBJ_cisd+=$(SRC_cisd:.cpp=.o)
OBJ_hci+=$(SRC_hci:.cpp=.o)
OBJ_forcyrus+=$(SRC_forcyrus:.cpp=.o)
OBJ_Excitations+=$(SRC_Excitations:.cpp=.o)
OBJ_hci2+=$(SRC_hci2:.cpp=.o)

.cpp.o :
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@


all: HCI stats forcyrus

boost/lib/libboost_serialization.a:   $(BOOST_SERIALIZATION_OBJ)
	ar r boost/lib/libboost_serialization.a $(BOOST_SERIALIZATION_OBJ)

boost/lib/libboost_mpi.a:             $(BOOST_MPI_OBJ)
	 ar r boost/lib/libboost_mpi.a $(BOOST_MPI_OBJ)

stats: stats.o
	$(CXX) -O3 stats.cpp -o stats
HCI	: $(OBJ_hci) boost/lib/libboost_serialization.a boost/lib/libboost_mpi.a
	$(CXX)   $(FLAGS) $(OPT) -o  HCI $(OBJ_hci) $(LFLAGS)
forcyrus	: $(OBJ_forcyrus) boost/lib/libboost_serialization.a boost/lib/libboost_mpi.a
	$(CXX)   $(FLAGS) $(OPT) -o  forcyrus $(OBJ_forcyrus) $(LFLAGS)
Excitations	: $(OBJ_Excitations)
	$(CXX)   $(FLAGS) $(OPT) -o  Excitations $(OBJ_Excitations) $(LFLAGS)
HCI2	: $(OBJ_hci2)
	$(CXX)   $(FLAGS) $(OPT) -o  HCI2 $(OBJ_hci2) $(LFLAGS)
CISD	: $(OBJ_cisd)
	$(CXX)   $(FLAGS) $(OPT) -o  CISD $(OBJ_cisd) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm CIST HCI

