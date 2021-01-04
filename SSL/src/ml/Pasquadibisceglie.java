package ml;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XLifecycleExtension;
import org.deckfour.xes.extension.std.XTimeExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;

import parser.Parser;
import reducer.ReducedLogContainer;

public class Pasquadibisceglie implements Approach {

	@Override
	public void createInputFiles(XLog referenceLog, Set<ReducedLogContainer> logs, String path, boolean lifecycle) {		
		//Extract and numerate activities
		Parser p = new Parser();
		Set<String> activities = new HashSet<String>();
		if(lifecycle == false) {
			activities = p.getActivities(referenceLog);
		}
		else {
			activities = p.getActivitiesWithLifecycle(referenceLog);
		}
				
		Map<String, Integer> activityIndexMap = new HashMap<String, Integer>();
		int index = 1;
		for(String activity : activities) {
			activityIndexMap.put(activity, index);
			index++;
		}
		
		//Transform Logs
		for(ReducedLogContainer container : logs) {
			transform(activityIndexMap, container.getLog(), container.getTitle(), path, lifecycle);
		}
		
	}
	
	private void transform(Map<String, Integer> activities, XLog log, String title, String path, boolean lifecycle) {
		StringBuilder sb = new StringBuilder();
		//Header of the CSV File
		sb.append("CaseID,Activity,Timestamp");
		sb.append("\n");
		
		//Transform Traces
		int counter = 1;
		Iterator<XTrace> logIterator = log.iterator();
		while(logIterator.hasNext()) {
			XTrace currentTrace = logIterator.next();
			
			Iterator<XEvent> traceIterator = currentTrace.iterator();
			while(traceIterator.hasNext()) {
				sb.append(counter);
				sb.append(",");
				XEvent currentEvent = traceIterator.next();
				//Activities
				if(lifecycle == true) {
					String activity = currentEvent.getAttributes().get(XConceptExtension.KEY_NAME).toString();
					String transition = currentEvent.getAttributes().get(XLifecycleExtension.KEY_TRANSITION).toString();
					
					sb.append(activities.get(activity+"-"+transition));
					sb.append(",");
				}
				else {
					sb.append(currentEvent.getAttributes().get(XConceptExtension.KEY_NAME));
				}
				
				//Timestamp
				Date extractedDate = XTimeExtension.instance().extractTimestamp(currentEvent);
				sb.append(extractedDate.getTime()/86400000.0);
				sb.append("\n");
			}
			counter++;
		}
		
		//Serializing
		File inputFile = new File(path+"\\inp_"+title+".csv");
		if(inputFile.exists()) {
			inputFile.delete();
		}
		inputFile = new File(path+"\\inp_"+title+".csv");
				
		try {
			FileWriter fileWriter = new FileWriter(inputFile, true);
			BufferedWriter writer = new BufferedWriter(fileWriter);
			PrintWriter out = new PrintWriter(writer);
			out.print(sb.toString());
			out.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

}
