"""This module is related to GUI class"""
import os
import sys

from PyQt5 import QtWidgets, QtCore

cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(cur_dir)
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class CustomDialog(QDialog):
    """
    This class is used to handle the dialog box that is presented to the Human to choose his preference.
    """

    def __init__(self, center_widget, left_branch, right_branch, parent=None):
        """
            Function: constructor
            Description: Initializes the class object attributes with initial values
            Inputs:
                -center_widget: Center Widget instance
                -left_branch: East branch to show east branch node attributes
                -right_branch: West branch to show west branch node attributes
                -parent: Parent widget if any, default value is None
            Output:
                None
        """
        super().__init__(parent)

        self.setWindowTitle("SNEAK Preference")
        self.buttonBox = QDialogButtonBox(center_widget)
        self.buttonBox.addButton("Option A", QDialogButtonBox.YesRole)
        self.buttonBox.addButton("Option B", QDialogButtonBox.NoRole)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.hor_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(QLabel("--------------------Option A--------------------"))
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(QLabel("--------------------Option B--------------------"))
        for item in left_branch:
            if item and len(item) != item.count(" "):
                temp_label = QLabel("* " + item)
                self.left_layout.addWidget(temp_label)
            elif item:
                temp_label = QLabel(item)
                self.left_layout.addWidget(temp_label)
        for item in right_branch:
            if item and len(item) != item.count(" "):
                temp_label = QLabel("* " + item)
                self.right_layout.addWidget(temp_label)
            elif item:
                temp_label = QLabel(item)
                self.right_layout.addWidget(temp_label)
        message = QLabel("Please select the option that suits you better")
        self.layout.addWidget(message)
        self.hor_layout.addLayout(self.left_layout)
        self.hor_layout.addLayout(self.right_layout)
        self.layout.addLayout(self.hor_layout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class UIHelper(QMainWindow):
    """
    This class is used to create Graphical User Interface which can be used for human interaction purposes.
    """
    current_active_widget = None

    def __init__(self, q_app_ref, whun_run, file_names, eval_files, is_oracle_enabled, parent=None):
        """
            Function: constructor
            Description: Initializes the class object attributes with initial values
            Inputs:
                -q_app_ref: QApplication reference
                -whun_run: Whun method to be invoked on clicking run button in GUI
                -parent: Parent widget if any, default value is None
            Output:
                None
        """
        super(UIHelper, self).__init__(parent)

        self.result_label = QLabel("Result")
        self.q_app_ref = q_app_ref
        self.file_names = file_names
        self.eval_files = eval_files
        self.is_oracle_enabled = is_oracle_enabled

        # Main Window
        self.setGeometry(0, 0, 1600, 800)
        self.setWindowTitle("SNEAK")

        # Central Widget
        self.central_widget = QWidget()
        self.layout_for_widget = QStackedLayout()

        # All screens that we need
        self.landing_widget = self.prepare_landing_screen(whun_run)
        self.landing_widget.show()
        self.wait_widget = self.prepare_wait_screen()
        self.wait_widget.hide()
        self.result_widget = self.prepare_results_screen()
        self.result_widget.hide()

        # Layout container for all widgets
        self.layout_for_widget.addWidget(self.landing_widget)
        self.layout_for_widget.addWidget(self.wait_widget)
        self.layout_for_widget.addWidget(self.result_widget)

        # Parent Layout
        self.central_widget.setLayout(self.layout_for_widget)

        # Set parent layout
        self.setCentralWidget(self.central_widget)

        self.landing_widget.show()
        self.result_widget.hide()
        self.wait_widget.hide()

        # Update current widget to Landing Screen
        self.current_active_widget = "LANDING"

    def prepare_landing_screen(self, whun_run):
        """
            Function: prepare_landing_screen
            Description: Function to initialize all the UI components of Landing Screen
            Inputs:
                -whun_run: Method to initiate SNEAK algorithm execution. Will be invoked on clicking run button in GUI
            Output:
                None
        """
        # Widget creation
        widget_obj = QWidget()

        # Widget properties initialization
        # self.wid1.setStyleSheet("""background: black;""")
        widget_obj.setGeometry(0, 0, 1600, 800)

        # Layout Creation
        layout = QVBoxLayout()

        # Header Label Creation
        header_label = QLabel()
        header_label.setText("Run SNEAK and find a configuration that best fits your requirements")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont('Arial', 25))
        header_label.setWordWrap(True)

        # Start Button Creation
        start_button = QPushButton("RUN", self)
        start_button.setFont(QFont('Arial', 22))
        start_button.clicked.connect(lambda: self.run_button_handler(whun_run))

        # Adding elements into layout and widget
        layout.addWidget(header_label)
        layout.addWidget(start_button)
        widget_obj.setLayout(layout)
        return widget_obj

    def prepare_wait_screen(self):
        """
            Function: prepare_wait_screen
            Description: Function to initialize all the UI components of Wait/Processing Screen
            Inputs:
                None
            Output:
                None
        """
        # Widget creation
        widget_obj = QWidget()

        # Widget properties initialization
        # self.wid1.setStyleSheet("""background: black;""")
        widget_obj.setGeometry(0, 0, 1600, 800)

        # Layout Creation
        layout = QVBoxLayout()

        # Header Label Creation
        header_label = QLabel()
        header_label.setText("Please wait while SNEAK runs in background")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setFont(QFont('Arial', 25))
        header_label.setWordWrap(True)

        # Adding elements into layout and widget
        layout.addWidget(header_label)
        widget_obj.setLayout(layout)
        return widget_obj

    def prepare_results_screen(self):
        """
            Function: prepare_results_screen
            Description: Function to initialize all the UI components of Iteration Screen
            Inputs:
                None
            Output:
                None
        """
        # Widget creation

        widget_obj = QtWidgets.QWidget()
        widget_obj.setGeometry(0, 0, 1600 , 800)

        layout = QtWidgets.QHBoxLayout(widget_obj)

        scrollArea = QtWidgets.QScrollArea(widget_obj)
        layout.addWidget(scrollArea)

        scrollAreaWidgetContents = QtWidgets.QWidget()
        scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1600, 3000))

        scrollArea.setWidget(scrollAreaWidgetContents)

        layout = QtWidgets.QHBoxLayout(scrollAreaWidgetContents)

        # Layout Creation
        # layout = QVBoxLayout()

        # Header Label Creation
        header_label = QLabel()
        header_label.setText("Thanks for using SNEAK! To the right are two results that SNEAK has come up with based on your preferences. Please rate the quality of each solution with respect to your requirements between 0 and 5\n(0 being the worst, 5 being the best).")
        header_label.setAlignment(Qt.AlignTop)
        header_label.setFont(QFont('Arial', 25))
        header_label.setWordWrap(True)

        #Configure Result Label
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setFont(QFont('Arial', 14))
        self.result_label.setWordWrap(True)

        # Adding elements into layout and widget
        layout.addWidget(header_label)
        layout.addWidget(self.result_label)
        widget_obj.setLayout(layout)
        return widget_obj

    def update_widget(self, next_widget=None):
        """
            Function: update_widget
            Description: Function to change screens based on the flow of the code
            Inputs:
                -next_widget: Value of the next screen that needes to be shown
            Output:
                None
        """
        if next_widget == "WAIT_SCREEN":
            self.landing_widget.hide()
            self.result_widget.hide()
            self.wait_widget.show()
            self.current_active_widget = "WAIT_SCREEN"
        elif next_widget == "ITERATION":
            self.landing_widget.hide()
            self.wait_widget.hide()
            self.result_widget.show()
            self.current_active_widget = "ITERATION"

    def run_button_handler(self, whun_run):
        """
            Function: run_button_handler
            Description: Function to handle the run button click on GUI
            Inputs:
                -whun_run: SNEAK method to be invoked on clicking run button in GUI
            Output:
                None
        """
        self.update_widget("WAIT_SCREEN")
        whun_run(self.file_names, self.eval_files, self.is_oracle_enabled)

    def update_result_label(self, result):
        """
            Function: update_result_label
            Description: Function to update Result label in the GUI
            Inputs:
                -result: String that contains list of prefernces that SNEAK has generated based on Human interaction
            Output:
                None
        """
        self.result_label.setText(result)

    def show_options_dialog(self, left_branch, right_branch):
        """
            Function: show_options_dialog
            Description: Function to show dialog box to the human inorder for the human to select his preferences
            Inputs:
                -left_branch: East branch nodes that contain attributes
                -right_branch: West branch nodes that contain attributes
            Output:
                - 1: Return 1 if the user has selected the attributes of the east branch
                - 0: Return 0 if the user has selected the attributes of the west branch
        """
        dlg = CustomDialog(self.central_widget, left_branch, right_branch)
        if dlg.exec():
            return 1
        else:
            return 0

