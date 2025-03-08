import pygame
import sys
import os
from scripts.capture_data import main as capture_main
from scripts.preprocess_data import main as preprocess_main
from scripts.train_model import main as train_main
from scripts.detect_posture import main as detect_main

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.is_hovered = False
        
    def draw(self, surface, font):
        pygame.draw.rect(surface, self.current_color, self.rect, border_radius=12)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            self.current_color = self.hover_color if self.is_hovered else self.color
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False

def main():
    pygame.init()
    
    # Set up display
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Posture Detection System")
    
    # Colors
    BACKGROUND = (36, 36, 36)
    BUTTON_COLOR = (52, 152, 219)
    BUTTON_HOVER = (41, 128, 185)
    
    # Font
    font = pygame.font.Font(None, 36)
    title_font = pygame.font.Font(None, 48)
    
    # Create buttons
    button_width = 200
    button_height = 50
    button_margin = 30
    start_y = HEIGHT // 2 - (button_height * 2 + button_margin * 1.5)
    
    buttons = {
        "capture": Button(WIDTH//2 - button_width//2, start_y, 
                         button_width, button_height, "Capture Data", 
                         BUTTON_COLOR, BUTTON_HOVER),
        "preprocess": Button(WIDTH//2 - button_width//2, start_y + button_height + button_margin, 
                            button_width, button_height, "Preprocess Data", 
                            BUTTON_COLOR, BUTTON_HOVER),
        "train": Button(WIDTH//2 - button_width//2, start_y + (button_height + button_margin) * 2, 
                       button_width, button_height, "Train Model", 
                       BUTTON_COLOR, BUTTON_HOVER),
        "detect": Button(WIDTH//2 - button_width//2, start_y + (button_height + button_margin) * 3, 
                        button_width, button_height, "Detect Posture", 
                        BUTTON_COLOR, BUTTON_HOVER)
    }
    
    # Status message
    status_message = ""
    status_color = (255, 255, 255)
    
    running = True
    while running:
        screen.fill(BACKGROUND)
        
        # Draw title
        title_surface = title_font.render("Posture Detection System", True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(WIDTH//2, 100))
        screen.blit(title_surface, title_rect)
        
        # Draw buttons
        for button in buttons.values():
            button.draw(screen, font)
        
        # Draw status message
        if status_message:
            status_surface = font.render(status_message, True, status_color)
            status_rect = status_surface.get_rect(center=(WIDTH//2, HEIGHT - 50))
            screen.blit(status_surface, status_rect)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            # Handle button events
            if buttons["capture"].handle_event(event):
                try:
                    print("Starting data capture...")
                    print("Use 'n' to switch between postures:")
                    print("- good_sitting")
                    print("- bad_sitting")
                    print("- sitting_forward")
                    print("- sitting_leanback")
                    print("- sitting_left")
                    print("- sitting_right")
                    capture_main()
                    status_message = "Data capture completed!"
                    status_color = (0, 255, 0)
                except Exception as e:
                    status_message = f"Error: {str(e)}"
                    status_color = (255, 0, 0)
                    
            elif buttons["preprocess"].handle_event(event):
                try:
                    preprocess_main()
                    status_message = "Data preprocessing completed!"
                    status_color = (0, 255, 0)
                except Exception as e:
                    status_message = f"Error: {str(e)}"
                    status_color = (255, 0, 0)
                    
            elif buttons["train"].handle_event(event):
                try:
                    training_report = train_main()
                    if training_report is not None:
                        status_message = f"Training completed! Final accuracy: {training_report['final_test_acc']:.2f}%"
                        status_color = (0, 255, 0)
                    else:
                        status_message = "Training failed! Check console for details."
                        status_color = (255, 0, 0)
                except Exception as e:
                    status_message = f"Error: {str(e)}"
                    status_color = (255, 0, 0)
                    
            elif buttons["detect"].handle_event(event):
                try:
                    detect_main()
                    status_message = "Detection completed!"
                    status_color = (0, 255, 0)
                except Exception as e:
                    status_message = f"Error: {str(e)}"
                    status_color = (255, 0, 0)
        
        pygame.display.flip()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()