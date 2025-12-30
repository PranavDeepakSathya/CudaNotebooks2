import dearpygui.dearpygui as dgp

def main(): 
  dgp.create_context()
  with dgp.window(label = "Hello DearPyGui"):
    dgp.add_text("this is a window")
    dgp.create_viewport(title='example', width = 1000, height = 400)
    dgp.setup_dearpygui()
    dgp.show_viewport()
    dgp.start_dearpygui()
    dgp.destroy_context()
    
main()