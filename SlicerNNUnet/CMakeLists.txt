#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerNNUNet)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/InstallLogic.py
  ${MODULE_NAME}Lib/Parameter.py
  ${MODULE_NAME}Lib/SegmentationLogic.py
  ${MODULE_NAME}Lib/Signal.py
  ${MODULE_NAME}Lib/Widget.py
  Testing/__init__.py
  Testing/InstallLogicTestCase.py
  Testing/IntegrationTestCase.py
  Testing/SegmentationLogicTestCase.py
  Testing/Utils.py
  Testing/WidgetTestCase.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/start_icon.png
  Resources/Icons/stop_icon.png
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
