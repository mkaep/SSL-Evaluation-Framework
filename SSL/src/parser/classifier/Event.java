package parser.classifier;


public class Event {
	private String eventAttributes;
	
	public Event(String eventAttributes) {
		this.eventAttributes = eventAttributes;
	}
	
	public String getEventAttributes() {
		return eventAttributes;
	}
	
	public void setEventAttributes(String eventAttributes) {
		this.eventAttributes = eventAttributes;
	}
	
	public String toString() {
		return eventAttributes;
	}
	
	@Override
	public boolean equals(Object o) {
		if(!(o instanceof Event)) {
			return false;
		}
		else {
			Event c = (Event) o;
			return (eventAttributes.equals(c.getEventAttributes()));
		}
	}

	@Override
	public int hashCode() {
		return 0;
	}

}
