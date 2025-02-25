class Event:
      def __init__(self, event_type):
            self.event_type = event_type.lower()

      def identify_event_type(self):
            if self.event_type == "top":
                  return "Top Event"
            elif self.event_type == "intermediate":
                  return "Intermediate Event"
            elif self.event_type == "basic":
                  return "Basic Event"
            else:
                  return "Unknown Event Type"

# Example usage
event = Event("top")
print(event.identify_event_type())  # Output: Top Event

