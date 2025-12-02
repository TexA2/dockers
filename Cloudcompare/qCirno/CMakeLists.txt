option( PLUGIN_STANDARD_QCIRNO "Install qCirno plugin" OFF )

if( PLUGIN_STANDARD_QCIRNO )
	project( QCIRNO_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )

	add_subdirectory( include )
	add_subdirectory( src )

	target_sources( ${PROJECT_NAME}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}/qCirno.qrc
	)

	find_package(Eigen3 REQUIRED)
	find_package( PCL 1.9.1 REQUIRED )
	include_directories(${EIGEN_INCLUDE_DIR})
	if( PCL_FOUND)
		# [C/C++]>[General]>[Additional Include Directories]
		include_directories( ${PCL_INCLUDE_DIRS} )

		# [C/C++]>[Preprocessor]>[Preprocessor Definitions]
		add_definitions( ${PCL_DEFINITIONS} )

		# For Use Not PreCompiled Features 
		#add_definitions( -DPCL_NO_PRECOMPILE )

		# [Linker]>[General]>[Additional Library Directories]
		link_directories( ${PCL_LIBRARY_DIRS} )

		# [Linker]>[Input]>[Additional Dependencies]
		target_link_libraries( QCIRNO_PLUGIN ${PCL_LIBRARIES} Eigen3::Eigen)
	endif()
endif()
