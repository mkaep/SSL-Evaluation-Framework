package parser;

import java.util.ArrayList;
import java.util.List;

import parser.classifier.Event;
/**
 * This class defines the concept of a trace variant. Traces are considered as equal with regard to different criteria. The Classifier defines
 * one or more process perspectives that our considered for determining the trace variant.
 * 
 * @author 	Martin Kaeppel
 * @version	18.12.2020
 * 
 */
public class TraceVariant {
	List<Event> events = new ArrayList<Event>();
	
	public TraceVariant() {
		
	}
	
	public void addEvent(Event event) {
		events.add(event);
	}
	
	public List<Event> getEvents() {
		return events;
	}
	
	@Override
	public boolean equals(Object o) {
		if(!(o instanceof TraceVariant)) {
			return false;
		}
		else {
			TraceVariant c = (TraceVariant) o;
			return (events.equals(c.getEvents()));
		}
	}

	@Override
	public int hashCode() {
		return 0;
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < events.size(); i++) {
			sb.append(events.get(i));
			if(i != events.size()-1) {
				sb.append("\t-->\t");
			}
		}
		return sb.toString();
	}

}
